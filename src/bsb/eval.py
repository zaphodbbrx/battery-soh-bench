"""Метрики и разбивки (ТЗ §10).

Основные:

* ``MAE`` в процентных пунктах SOH;
* ``P95`` абсолютной ошибки — для BMS худший случай важнее среднего;
* разбивка по химии, температуре и децилям SOH (интересует поведение на сильно
  состаренных ячейках, их всегда меньше).

Номинация «калибровка»: покрытие 90% интервала и средняя ширина интервала.

Вывод — таблица в stdout и ``results.json`` (ТЗ §10).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from bsb.config import BenchmarkConfig
from bsb.store import read_cells

__all__ = [
    "evaluate",
    "load_truth",
    "render_report",
    "run_eval",
    "save_results",
]

#: Доля, по которой считается «худший случай» в P95.
P95 = 0.95


def load_truth(config: BenchmarkConfig) -> pl.DataFrame:
    """Эталон теста: метки из файла организатора плюс метаданные ячеек (§4.1).

    Участникам этот файл не выдаётся: ``test_unlabeled.parquet`` не содержит
    ни ``soh``, ни ``cell_id``.
    """
    path = config.paths.fragments_path("test", labeled=True)
    if not path.is_file():
        raise FileNotFoundError(
            f"{path}: эталон теста не сгенерирован (bsb fragments --split test; фаза 7)"
        )
    fragments = pl.read_parquet(path)
    cells = read_cells(config).select("cell_id", "chemistry", "temperature_setpoint_c", "dataset")
    return fragments.join(cells, on="cell_id", how="left")


def _metrics(
    frame: pl.DataFrame, *, truth_column: str = "soh", pred_column: str = "soh_pred"
) -> dict[str, Any]:
    """MAE, P95 абсолютной ошибки и смещение в процентных пунктах."""
    if frame.height == 0:
        return {"n": 0, "mae_pp": None, "p95_abs_pp": None, "bias_pp": None}
    error = (frame[pred_column] - frame[truth_column]).to_numpy()
    absolute = np.abs(error)
    return {
        "n": int(error.size),
        "mae_pp": round(float(absolute.mean()) * 100.0, 4),
        "p95_abs_pp": round(float(np.quantile(absolute, P95)) * 100.0, 4),
        "bias_pp": round(float(error.mean()) * 100.0, 4),
    }


def _group_metrics(frame: pl.DataFrame, key: str, *, sort: str = "value") -> list[dict[str, Any]]:
    """Метрики по группам одной колонки.

    ``sort``: ``value`` — по значению группы (химия, температура),
    ``mae`` — худшие сверху, ``decile`` — по возрастанию SOH, чтобы сильно
    состаренные ячейки были в начале таблицы (§10).
    """
    if frame.height == 0:
        return []
    rows: list[dict[str, Any]] = []
    for value, part in frame.group_by(key, maintain_order=True):
        rows.append({key: value[0] if isinstance(value, tuple) else value, **_metrics(part)})
    if sort == "mae":
        return sorted(rows, key=lambda item: (item["mae_pp"] is None, -(item["mae_pp"] or 0.0)))
    return sorted(rows, key=lambda item: str(item[key]))


def _add_soh_deciles(frame: pl.DataFrame, deciles: int) -> pl.DataFrame:
    """Разбить истинный SOH на децили — поведение на состаренных ячейках (§10)."""
    if frame.height < deciles:
        return frame.with_columns(pl.lit("все", dtype=pl.String).alias("soh_decile"))
    soh = frame["soh"].to_numpy()
    edges = np.quantile(soh, np.linspace(0.0, 1.0, deciles + 1))
    edges = np.maximum.accumulate(edges)
    inner = edges[1:-1]
    index = np.clip(np.digitize(soh, inner, right=False), 0, deciles - 1)
    labels = np.array([f"{edges[i]:.3f}–{edges[i + 1]:.3f}" for i in range(deciles)])
    return frame.with_columns(pl.Series("soh_decile", labels[index], dtype=pl.String))


def _calibration(frame: pl.DataFrame) -> dict[str, Any] | None:
    """Покрытие и средняя ширина интервала (§10, номинация «калибровка»)."""
    if "soh_lo" not in frame.columns or "soh_hi" not in frame.columns:
        return None
    usable = frame.filter(frame["soh_lo"].is_not_null() & frame["soh_hi"].is_not_null())
    if usable.height == 0:
        return None
    covered = ((usable["soh_lo"] <= usable["soh"]) & (usable["soh"] <= usable["soh_hi"])).mean()
    width = (usable["soh_hi"] - usable["soh_lo"]).mean()
    return {
        "n": int(usable.height),
        "coverage": round(float(covered), 4),
        "mean_width_pp": round(float(width) * 100.0, 4),
    }


def evaluate(
    submission: pl.DataFrame,
    truth: pl.DataFrame,
    *,
    config: BenchmarkConfig,
    hard_mode: bool = False,
) -> dict[str, Any]:
    """Посчитать метрики сабмита против эталона (ТЗ §10).

    Parameters
    ----------
    submission
        Кадр с ``fragment_id``, ``soh_pred`` и, возможно, ``soh_lo`` / ``soh_hi``.
    truth
        Эталон из :func:`load_truth`.
    hard_mode
        ``True`` — бонусная номинация LFP (§9); ``False`` — основной тест
        NCA + NMC.
    """
    chemistries = (
        config.eval.hard_mode_chemistries if hard_mode else config.eval.main_test_chemistries
    )
    # Основной тест — только NCA и NMC (§9); LFP считается отдельной номинацией,
    # поэтому и границы теста, и полнота сабмита считаются по выбранным химиям.
    mode_truth = truth.filter(pl.col("chemistry").is_in(list(chemistries)))
    joined = mode_truth.join(submission, on="fragment_id", how="inner")
    selected = joined
    mode_ids = set(mode_truth["fragment_id"].to_list())
    all_ids = set(truth["fragment_id"].to_list())
    submission_ids = set(submission["fragment_id"].to_list())
    # «Не хватает» — это про оцениваемые химии: без них метрика не посчитается.
    # «Лишние» — только то, чего нет в тестовом файле вовсе: фрагменты LFP в
    # основном режиме не лишние, они просто не входят в эту номинацию (§9).
    missing = sorted(mode_ids - submission_ids)
    extra = sorted(submission_ids - all_ids)

    result: dict[str, Any] = {
        "mode": "hard" if hard_mode else "main",
        "chemistries": list(chemistries),
        "n_test_file": truth.height,
        "n_truth": mode_truth.height,
        "n_submission": submission.height,
        "n_matched": joined.height,
        "n_missing": len(missing),
        "n_extra": len(extra),
        "coverage_of_test": (
            round(joined.height / mode_truth.height, 4) if mode_truth.height else None
        ),
        "overall": _metrics(selected),
        "by_chemistry": _group_metrics(selected, "chemistry"),
        "by_temperature_setpoint_c": _group_metrics(selected, "temperature_setpoint_c"),
        "by_soh_decile": _group_metrics(
            _add_soh_deciles(selected, config.eval.soh_deciles), "soh_decile", sort="decile"
        ),
        "calibration": _calibration(selected),
    }
    if missing:
        result["missing_examples"] = missing[:10]
    if extra:
        result["extra_examples"] = extra[:10]
    return result


def _format_group(rows: list[dict[str, Any]], key: str, title: str) -> list[str]:
    """Отформатировать разбивку для stdout."""
    if not rows:
        return []
    lines = ["", title, f"{key:>22} {'n':>6} {'MAE, п.п.':>10} {'P95, п.п.':>10} {'смещение':>10}"]
    for row in rows:
        lines.append(
            f"{row[key]!s:>22} {row['n']:>6} {row['mae_pp']:>10.3f} "
            f"{row['p95_abs_pp']:>10.3f} {row['bias_pp']:>10.3f}"
        )
    return lines


def render_report(result: dict[str, Any]) -> str:
    """Собрать таблицу метрик для stdout (ТЗ §10)."""
    overall = result["overall"]
    lines = [
        f"режим: {result['mode']} ({', '.join(result['chemistries'])})",
        f"сабмит: {result['n_submission']} строк, тест: {result['n_test_file']} фрагментов, "
        f"оценивается: {result['n_truth']}, сопоставлено: {result['n_matched']} "
        f"(покрытие {result['coverage_of_test']}), "
        f"не хватает: {result['n_missing']}, лишних: {result['n_extra']}",
        "",
        f"MAE:  {overall['mae_pp']} п.п.   P95: {overall['p95_abs_pp']} п.п.   "
        f"смещение: {overall['bias_pp']} п.п.   n = {overall['n']}",
    ]
    lines += _format_group(result["by_chemistry"], "chemistry", "по химии:")
    lines += _format_group(
        result["by_temperature_setpoint_c"], "temperature_setpoint_c", "по температуре:"
    )
    lines += _format_group(result["by_soh_decile"], "soh_decile", "по децилям SOH:")
    calibration = result["calibration"]
    lines.append("")
    if calibration is None:
        lines.append("калибровка: интервалы не заданы — номинация недоступна (ТЗ §11)")
    else:
        lines.append(
            f"калибровка: покрытие {calibration['coverage']:.3f} "
            f"(номинал {result.get('nominal_coverage', 0.9)}), "
            f"средняя ширина {calibration['mean_width_pp']:.2f} п.п., n = {calibration['n']}"
        )
    return "\n".join(lines)


def save_results(result: dict[str, Any], path: str | Path) -> Path:
    """Записать ``results.json`` (ТЗ §10)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    return out


def run_eval(
    config: BenchmarkConfig,
    submission: pl.DataFrame,
    *,
    hard_mode: bool = False,
    results_path: str | Path | None = None,
) -> dict[str, Any]:
    """Посчитать метрики и записать ``results.json``.

    Размер файла сабмита (номинация «экономность», §10) добавляется вызывающей
    стороной — здесь он неизвестен, потому что кадр уже в памяти.
    """
    truth = load_truth(config)
    result = evaluate(submission, truth, config=config, hard_mode=hard_mode)
    result["nominal_coverage"] = config.eval.calibration_coverage
    save_results(result, results_path or config.paths.results_path)
    return result
