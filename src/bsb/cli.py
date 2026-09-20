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
from typing import Any

import polars as pl

from bsb import __version__
from bsb.baseline import MODEL_NAMES, fit_predict, noise_floor, oracle_constant_mae
from bsb.config import BenchmarkConfig, ConfigError, load_config
from bsb.dataset import iter_bundles, read_split_cell_ids
from bsb.eval import evaluate, load_truth, render_report, run_eval, save_results
from bsb.fragments import generate_fragments, strip_labels
from bsb.ingest import ADAPTERS, get_adapter
from bsb.labels import NoReferenceCycles
from bsb.schema import SchemaViolation, validate_bundle, validate_fragments
from bsb.splits import CellLeakageError, assert_no_cell_leakage, load_splits, split_path
from bsb.store import (
    clear_dataset_parts,
    compact,
    list_cells,
    parts_dir,
    read_cells,
    read_cycles,
    write_bundle,
)
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
#: Команды, ещё не реализованные, с фазой плана §15.
_PENDING: dict[str, str] = {
    "labels": "фаза 2",
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
    compact_parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="разрешить запись унифицированного уровня без остальных датасетов",
    )

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
    baseline_parser.add_argument("--model", default="elastic_net", choices=MODEL_NAMES)
    baseline_parser.add_argument("--out", type=Path, default=None, help="куда писать results.json")

    noise_parser = subparsers.add_parser(
        "noise-floor", help="оценить шумовой пол по репликам ячеек"
    )
    noise_parser.add_argument("--out", type=Path, default=None, help="куда писать JSON")

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
    compact(config, datasets=datasets, allow_partial=bool(getattr(args, "allow_partial", False)))
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
    if args.limit is None:
        # Полный прогон заменяет датасет целиком: ячейки, которых больше нет в
        # сырых данных или которые исключены правилом конфига, уходят из
        # хранилища. Прогон с --limit ничего не чистит: он по определению
        # частичный и не должен стирать остальное.
        clear_dataset_parts(config, args.dataset)
    print(f"{args.dataset}: найдено {len(sources)} ячеек в {raw_dir}")
    warnings: list[str] = []
    failures: list[tuple[str, str]] = []
    skipped: list[tuple[str, str]] = []
    for index, source in enumerate(sources, start=1):
        try:
            bundle = adapter.mark_reference_cycles(adapter.load_cell(source))
            warnings.extend(f"{source.cell_id}: {item}" for item in validate_bundle(bundle))
            write_bundle(config, bundle, dataset=args.dataset)
        except NoReferenceCycles as exc:
            # Запись без опорных циклов — не сбой разбора, а свойство данных:
            # она просто не даёт фрагментов (в архиве Sandia так устроены файлы
            # 45 °C — мелкие частичные циклы). Отчитываемся отдельно, чтобы
            # «не разобрано ячеек» не смешивало это с настоящими ошибками.
            skipped.append((source.cell_id, str(exc)))
            continue
        except Exception as exc:
            # Одна плохая ячейка не должна ронять разбор всего датасета:
            # причина попадает в отчёт, остальные ячейки обрабатываются.
            # Ловим широко сознательно: сырые наборы дают самые разные ошибки
            # разбора (битые файлы, чужие форматы), и останавливать из-за них
            # прогон на 15 ячейках бессмысленно.
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
    if skipped:
        print(
            f"пропущено записей без опорных циклов: {len(skipped)} "
            f"(например, только частичные циклы), первые: {[item[0] for item in skipped[:3]]}",
            file=sys.stderr,
        )
    if failures:
        print(f"не разобрано ячеек: {len(failures)} из {len(sources)}", file=sys.stderr)
    if args.compact:
        # Собираем все разобранные датасеты, а не только текущий: сборка
        # подмножества перезаписала бы cells.parquet без остальных.
        return _cmd_compact(config, argparse.Namespace(datasets=None))
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


def _fragment_metadata(config: BenchmarkConfig, fragments: pl.DataFrame) -> pl.DataFrame:
    """Метаданные ячейки для каждого фрагмента (нужны пробе §12.3).

    ``chemistry`` добавлена ради контроля ``chem_constant``: это единственное
    поле, которое участнику не выдаётся, поэтому модель с ней — не сабмит, а
    эталон для проверки на утечку.
    """
    cells = read_cells(config).select(
        "cell_id", "chemistry", "nominal_capacity_ah", "temperature_setpoint_c"
    )
    return (
        fragments.select("fragment_id", "cell_id")
        .join(cells, on="cell_id", how="left")
        .select("fragment_id", "chemistry", "nominal_capacity_ah", "temperature_setpoint_c")
    )


def _cmd_baseline(config: BenchmarkConfig, args: argparse.Namespace) -> int:
    """Команда ``baseline``: обучить модель и посчитать метрики (§12.5).

    Считаются обе номинации §9: основная (NCA + NMC) и бонусная (LFP) — по
    решению владельца это и есть два сравнения на SNL.

    Рядом с запрошенной моделью всегда считаются две тривиальные константы —
    ``constant`` (глобальное среднее обучения) и ``chem_constant`` (среднее по
    химии) — и печатается вердикт §12.3. Без второй константы проверка на
    утечку бессмысленна: тест смешивает химии, и «пол» проигрывает не из-за
    утечки позиции окна, а из-за межхимического сдвига SOH.
    """
    train = pl.read_parquet(config.paths.fragments_path("train", labeled=True))
    test = pl.read_parquet(config.paths.fragments_path("test", labeled=True))
    metadata = _fragment_metadata(config, pl.concat([train, test], how="vertical"))
    truth = load_truth(config)

    models = [args.model, *(m for m in ("constant", "chem_constant") if m != args.model)]
    submissions = {
        model: fit_predict(train, test, model=model, config=config, metadata=metadata)
        for model in models
    }
    by_mode = {
        model: {
            mode: evaluate(sub, truth, config=config, hard_mode=hard)
            for mode, hard in (("main", False), ("hard", True))
        }
        for model, sub in submissions.items()
    }

    result: dict[str, object] = {"model": args.model, "references": {}}
    for mode, hard in (("main", False), ("hard", True)):
        metrics = by_mode[args.model][mode]
        result[mode] = metrics
        for model in models:
            if model != args.model:
                result["references"].setdefault(model, {})[mode] = by_mode[model][mode]
        chemistries = (
            config.eval.hard_mode_chemistries if hard else config.eval.main_test_chemistries
        )
        oracle = oracle_constant_mae(truth, chemistries)
        result["references"].setdefault("constant_test_domain", {})[mode] = round(oracle, 4)
        print(f"--- {mode} ({', '.join(metrics['chemistries'])})")
        print(render_report(metrics))
        print(_render_reference_block(by_mode, args.model, mode, chemistries, oracle))
        print()

    save_results(result, args.out or config.paths.results_path)
    print(f"results.json: {args.out or config.paths.results_path}")
    return EXIT_OK


def _render_reference_block(
    by_mode: dict[str, dict[str, dict[str, Any]]],
    model: str,
    mode: str,
    chemistries: Sequence[str],
    oracle_mae: float,
) -> str:
    """Сравнить модель с тривиальными константами и вынести вердикт §12.3.

    Нулём для пробы служит не среднее обучения, а константа тест-домена:
    обучение (Barkholtz, почти свежие ячейки) смещено относительно SNL, и проба
    обходит «пол» за счёт компенсации этого сдвига, а не за счёт положения окна.
    """
    rows = {name: by_mode[name][mode]["overall"]["mae_pp"] for name in by_mode}
    best_trivial = min(rows["constant"], rows["chem_constant"])
    lines = [
        "опорные константы (MAE, п.п.):",
        f"  constant (среднее обучения)     {rows['constant']:8.3f}",
        f"  chem_constant (среднее химии)   {rows['chem_constant']:8.3f}",
        f"  constant тест-домена [{', '.join(chemistries)}] {oracle_mae:6.3f}  ← ноль §12.3",
        f"  модель {model}: {rows[model]:.3f} — "
        + ("лучше" if rows[model] < best_trivial else "НЕ лучше")
        + " лучшей константы обучения"
        + (f" на {best_trivial - rows[model]:+.3f} п.п." if rows[model] < best_trivial else ""),
    ]
    if model == "probe":
        limit = oracle_mae / 1.15
        verdict = "пройдена" if rows["probe"] >= limit else "ПРОВАЛЕНА"
        lines.append(
            f"  утечка §12.3: проба {rows['probe']:.3f} против порога "
            f"{oracle_mae:.3f}/1.15 = {limit:.3f} — проверка {verdict}"
        )
    return "\n".join(lines)


def _cmd_noise_floor(config: BenchmarkConfig, args: argparse.Namespace) -> int:
    """Команда ``noise-floor``: оценить потолок по репликам (§12.4)."""
    cycles = read_cycles(config)
    cells = read_cells(config)
    result = noise_floor(cycles, cells, config)
    print(f"условий с репликами: {result['conditions']}, ячеек в них: {result['replicates']}")
    if result["noise_floor_pp"] is None:
        print(f"шумовой пол не посчитан: {result.get('note')}")
    else:
        print(f"шумовой пол: {result['noise_floor_pp']} п.п. (медиана разброса между ячейками)")
        print("самые шумные условия:")
        for item in result["per_condition"][:5]:
            print(f"   {item}")
    save_results(result, args.out or config.paths.results_path.with_name("noise_floor.json"))
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
            "baseline": lambda cfg: _cmd_baseline(cfg, args),
            "noise_floor": lambda cfg: _cmd_noise_floor(cfg, args),
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
