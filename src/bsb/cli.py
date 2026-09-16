"""Командная строка ``bsb`` (ТЗ §13).

Команды каркаса (``config-check``, ``splits-check``, ``compact``, ``ingest``)
работают уже в фазе 0; команды, реализуемые позже, честно падают с кодом 3 и
называют фазу плана §15, а не делают вид, что что-то посчитали.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from bsb import __version__
from bsb.config import BenchmarkConfig, ConfigError, load_config
from bsb.ingest import ADAPTERS, get_adapter
from bsb.schema import SchemaViolation, validate_bundle
from bsb.splits import CellLeakageError, assert_no_cell_leakage, load_splits, split_path
from bsb.store import compact, list_cells, parts_dir, write_bundle

__all__ = ["main"]

EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_NOT_IMPLEMENTED = 3

#: Команда → фаза плана §15, в которой она реализуется.
_PENDING: dict[str, str] = {
    "labels": "фаза 2",
    "fragments": "фаза 3",
    "noise-floor": "фаза 8",
    "baseline": "фаза 8",
    "eval": "фаза 7",
    "validate": "фаза 7",
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
