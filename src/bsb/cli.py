"""Командная строка ``bsb`` (ТЗ §13).

Реализованы ``config-check``, ``splits-check``, ``compact``, ``ingest``,
``fragments``, ``validate`` и ``eval``. Команды, реализуемые позже, честно
падают с кодом 3 и называют фазу плана §15, а не делают вид, что что-то
посчитали.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import polars as pl

from bsb import __version__
from bsb.config import BenchmarkConfig, ConfigError, load_config
from bsb.dataset import iter_bundles, read_split_cell_ids
from bsb.eval import render_report, run_eval, save_results
from bsb.fragments import generate_fragments, strip_labels
from bsb.ingest import ADAPTERS, get_adapter
from bsb.schema import SchemaViolation, validate_bundle, validate_fragments
from bsb.splits import CellLeakageError, assert_no_cell_leakage, load_splits, split_path
from bsb.store import compact, list_cells, parts_dir, write_bundle
from bsb.validate_submission import (
    expected_test_ids,
    read_submission,
    validate_submission,
)

__all__ = ["main"]

EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_NOT_IMPLEMENTED = 3

#: Команда → фаза плана §15, в которой она реализуется.
_PENDING: dict[str, str] = {
    "labels": "фаза 2",
    "noise-floor": "фаза 8",
    "baseline": "фаза 8",
}


def build_parser() -> argparse.ArgumentParser:
    """Собрать парсер командной строки."""
    parser = argparse.ArgumentParser(
        prog="bsb",
        description=(
            "battery-soh-bench: тулкит соревнования по оценке SOH по фрагменту зарядной кривой"
        ),
    )
    parser.add_argument("--config", type=Path, default=None, help="путь к benchmark.yaml")
    parser.add_argument("--version", action="version", version=f"bsb {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="КОМАНДА")

    subparsers.add_parser("config-check", help="проверить конфиг и напечатать сводку")
    subparsers.add_parser("splits-check", help="проверить отсутствие утечки ячеек между сплитами")

    compact_parser = subparsers.add_parser(
        "compact", help="собрать унифицированный уровень из parts/"
    )
    compact_parser.add_argument("--datasets", default=None, help="список датасетов через запятую")

    ingest_parser = subparsers.add_parser("ingest", help="разобрать сырые данные датасета")
    ingest_parser.add_argument("--dataset", required=True, choices=sorted(ADAPTERS))
    ingest_parser.add_argument("--raw", type=Path, default=None, help="каталог сырых данных")
    ingest_parser.add_argument("--limit", type=int, default=None, help="обработать первые N ячеек")
    ingest_parser.add_argument(
        "--compact", action="store_true", help="сразу собрать unified-уровень"
    )

    labels_parser = subparsers.add_parser("labels", help="разметить опорные циклы и посчитать soh")
    labels_parser.add_argument("--dataset", required=True, choices=sorted(ADAPTERS))

    fragments_parser = subparsers.add_parser("fragments", help="сгенерировать фрагменты сплита")
    fragments_parser.add_argument("--split", required=True, choices=("train", "val", "test"))
    fragments_parser.add_argument("--mode", default="fixed", choices=("fixed", "sampling"))
    fragments_parser.add_argument(
        "--cells", default=None, help="явный список cell_id через запятую (отладка)"
    )
    fragments_parser.add_argument(
        "--out", type=Path, default=None, help="куда писать (обязателен вместе с --cells)"
    )

    baseline_parser = subparsers.add_parser("baseline", help="посчитать бейзлайн")
    baseline_parser.add_argument(
        "--model", default="elastic_net", choices=("constant", "probe", "elastic_net")
    )

    subparsers.add_parser("noise-floor", help="оценить шумовой пол по репликам ячеек")

    eval_parser = subparsers.add_parser("eval", help="посчитать метрики сабмита")
    eval_parser.add_argument("--submission", type=Path, required=True)
    eval_parser.add_argument(
        "--hard-mode", action="store_true", help="бонусная номинация LFP (ТЗ §9)"
    )
    eval_parser.add_argument("--out", type=Path, default=None, help="куда писать results.json")

    validate_parser = subparsers.add_parser("validate", help="проверить формат сабмита")
    validate_parser.add_argument("--submission", type=Path, required=True)
    return parser


def _load(args: argparse.Namespace) -> BenchmarkConfig:
    """Загрузить конфиг с учётом ``--config``."""
    return load_config(args.config)


def _cmd_config_check(config: BenchmarkConfig) -> int:
    """Команда ``config-check``: напечатать сводку по конфигу."""
    fragment = config.fragment
    print(f"конфиг:      {config.source_path}")
    print(f"корень:      {config.root}")
    print(f"версия:      {config.version}, seed={config.seed}")
    print(
        f"разметка:    reference_only={config.labels.reference_only}, "
        f"allow_interpolated={config.labels.allow_interpolated}, "
        f"max_soh_gap_cycles={config.labels.max_soh_gap_cycles}"
    )
    print(
        f"фрагменты:   delta_v={fragment.delta_v} В, n_points={fragment.n_points}, "
        f"phase={fragment.phase}, v_normalization={fragment.v_normalization}, "
        f"channels={list(fragment.channels)}"
    )
    print(f"сплиты:      unit={config.splits.unit}, test={list(config.splits.test_datasets)}")
    print(
        f"eval:        main={list(config.eval.main_test_chemistries)}, "
        f"hard_mode={list(config.eval.hard_mode_chemistries)}"
    )
    print("конфиг корректен")
    return EXIT_OK


def _cmd_splits_check(config: BenchmarkConfig) -> int:
    """Команда ``splits-check``: проверить утечку ячеек между сплитами (ТЗ §8)."""
    splits = load_splits(config)
    for split in config.splits.split_names:
        print(f"{split:>5}: {len(splits[split]):4d} ячеек  ({split_path(config, split)})")
    try:
        assert_no_cell_leakage(splits)
    except CellLeakageError as exc:
        print(f"утечка: {exc}", file=sys.stderr)
        return EXIT_FAILURE
    print("утечки между сплитами нет")
    return EXIT_OK


def _cmd_compact(config: BenchmarkConfig, args: argparse.Namespace) -> int:
    """Команда ``compact``: собрать унифицированный уровень."""
    datasets = args.datasets.split(",") if args.datasets else None
    compact(config, datasets=datasets)
    cells = list_cells(config)
    print(f"унифицированный уровень собран: {len(cells)} ячеек в {config.paths.unified_dir}")
    return EXIT_OK


def _cmd_ingest(config: BenchmarkConfig, args: argparse.Namespace) -> int:
    """Команда ``ingest``: разобрать сырые данные датасета (ТЗ §6)."""
    adapter = get_adapter(args.dataset, config)
    raw_dir = args.raw or (config.paths.raw_dir / args.dataset)
    sources = adapter.discover(raw_dir)
    if args.limit is not None:
        sources = sources[: args.limit]
    if not sources:
        print(f"в {raw_dir} не найдено ни одной ячейки", file=sys.stderr)
        return EXIT_FAILURE
    print(f"{args.dataset}: найдено {len(sources)} ячеек в {raw_dir}")
    warnings: list[str] = []
    failures: list[tuple[str, str]] = []
    for index, source in enumerate(sources, start=1):
        try:
            bundle = adapter.mark_reference_cycles(adapter.load_cell(source))
            warnings.extend(f"{source.cell_id}: {item}" for item in validate_bundle(bundle))
            write_bundle(config, bundle, dataset=args.dataset)
        except (SchemaViolation, ValueError, KeyError) as exc:
            # Одна плохая ячейка не должна ронять разбор всего датасета:
            # причина попадает в отчёт, остальные ячейки обрабатываются.
            failures.append((source.cell_id, str(exc)))
            print(f"  [{index}/{len(sources)}] {source.cell_id}: ОШИБКА: {exc}", file=sys.stderr)
            continue
        print(
            f"  [{index}/{len(sources)}] {source.cell_id}: {bundle.cycles.height} циклов, "
            f"{bundle.timeseries.height} строк timeseries"
        )
    for warning in warnings:
        print(f"предупреждение: {warning}", file=sys.stderr)
    print(f"записано в {parts_dir(config, args.dataset)}")
    if failures:
        print(f"не разобрано ячеек: {len(failures)} из {len(sources)}", file=sys.stderr)
    if args.compact:
        return _cmd_compact(config, argparse.Namespace(datasets=args.dataset))
    return EXIT_OK


def _cmd_fragments(config: BenchmarkConfig, args: argparse.Namespace) -> int:
    """Команда ``fragments``: сгенерировать фрагменты сплита (ТЗ §7).

    Режим ``sampling`` намеренно не пишет артефакт: по §7 это аугментация на
    лету, и коммитить её вместо фиксированного набора нельзя.
    """
    if args.mode == "sampling":
        print(
            "режим sampling не создаёт артефакт: по ТЗ §7 это аугментация на лету. "
            "Используйте bsb.dataset.sample_fragments(...) в обучении.",
            file=sys.stderr,
        )
        return EXIT_FAILURE
    if args.cells and args.out is None:
        print(
            "--cells без --out запрещён: частичный список не должен выглядеть как сплит "
            "(файл сплита перезаписывать нельзя).",
            file=sys.stderr,
        )
        return EXIT_FAILURE

    if args.cells:
        cell_ids = [item.strip() for item in args.cells.split(",") if item.strip()]
    else:
        cell_ids = read_split_cell_ids(config, args.split)
        if not cell_ids:
            print(
                f"сплит {args.split} пуст: заполните {split_path(config, args.split)} "
                "(фаза 7 плана §15) или используйте --cells для отладки",
                file=sys.stderr,
            )
            return EXIT_FAILURE

    frames = []
    warnings: list[str] = []
    for bundle in iter_bundles(config, cell_ids):
        frame = generate_fragments(bundle, config)
        warnings.extend(
            f"{bundle.cell_id}: {item}"
            for item in validate_fragments(frame, labeled=True, n_points=config.fragment.n_points)
        )
        if frame.height == 0:
            print(f"  {bundle.cell_id}: ни одного окна не построено", file=sys.stderr)
        frames.append(frame)
    fragments = pl.concat(frames, how="vertical") if frames else pl.DataFrame()
    for warning in warnings:
        print(f"предупреждение: {warning}", file=sys.stderr)
    if fragments.height == 0:
        print(
            "фрагментов не построено: проверьте fragment.phase и min_points_in_window",
            file=sys.stderr,
        )
        return EXIT_FAILURE
    validate_fragments(fragments, labeled=True, n_points=config.fragment.n_points)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        labeled_path = args.out
        unlabeled_path = None
    else:
        labeled_path = config.paths.fragments_path(args.split, labeled=True)
        unlabeled_path = (
            config.paths.fragments_path(args.split, labeled=False) if args.split == "test" else None
        )
        labeled_path.parent.mkdir(parents=True, exist_ok=True)
    fragments.write_parquet(labeled_path)
    print(f"фрагментов: {fragments.height} ({len(cell_ids)} ячеек)")
    print(f"эталон:      {labeled_path}")
    if unlabeled_path is not None:
        test_frame = strip_labels(fragments)
        validate_fragments(test_frame, labeled=False, n_points=config.fragment.n_points)
        test_frame.write_parquet(unlabeled_path)
        print(f"участникам:  {unlabeled_path}")
    return EXIT_OK


def _cmd_validate(config: BenchmarkConfig, args: argparse.Namespace) -> int:
    """Команда ``validate``: проверить формат сабмита (ТЗ §11)."""
    report = validate_submission(
        args.submission, expected_ids=expected_test_ids(config), config=config
    )
    print(report.render())
    return EXIT_OK if report.ok else EXIT_FAILURE


def _cmd_eval(config: BenchmarkConfig, args: argparse.Namespace) -> int:
    """Команда ``eval``: посчитать метрики сабмита (ТЗ §10)."""
    report = validate_submission(
        args.submission, expected_ids=expected_test_ids(config), config=config
    )
    if not report.ok:
        print(report.render(), file=sys.stderr)
        print("метрики не считаются, пока сабмит не прошёл валидатор (ТЗ §11)", file=sys.stderr)
        return EXIT_FAILURE
    submission = read_submission(args.submission)
    result = run_eval(config, submission, hard_mode=args.hard_mode)
    result["submission_path"] = str(args.submission)
    result["submission_bytes"] = Path(args.submission).stat().st_size
    save_results(result, args.out or config.paths.results_path)
    print(render_report(result))
    print(f"\nresults.json: {args.out or config.paths.results_path}")
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    """Точка входа ``bsb``. Возвращает код выхода процесса."""
    parser = build_parser()
    args = parser.parse_args(argv)
    command = args.command.replace("-", "_")
    if args.command in _PENDING:
        phase = _PENDING[args.command]
        print(
            f"команда {args.command!r} ещё не реализована ({phase} плана §15 ТЗ)",
            file=sys.stderr,
        )
        return EXIT_NOT_IMPLEMENTED
    try:
        config = _load(args)
        handler = {
            "config_check": lambda cfg: _cmd_config_check(cfg),
            "splits_check": lambda cfg: _cmd_splits_check(cfg),
            "compact": lambda cfg: _cmd_compact(cfg, args),
            "ingest": lambda cfg: _cmd_ingest(cfg, args),
            "fragments": lambda cfg: _cmd_fragments(cfg, args),
            "validate": lambda cfg: _cmd_validate(cfg, args),
            "eval": lambda cfg: _cmd_eval(cfg, args),
        }[command]
        return handler(config)
    except (ConfigError, FileNotFoundError, SchemaViolation, CellLeakageError, KeyError) as exc:
        print(f"ошибка: {exc}", file=sys.stderr)
        return EXIT_FAILURE
    except NotImplementedError as exc:
        print(f"не реализовано: {exc}", file=sys.stderr)
        return EXIT_NOT_IMPLEMENTED


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
