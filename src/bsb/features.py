"""Признаки окна для бейзлайна (ТЗ §12.5).

Обязательный набор из §12.5:

* ``dQ/dV`` со сглаживанием Савицкого-Голея и статистики ``dQ/dV`` внутри окна;
* ``v_start`` / ``v_end`` / ширина окна;
* средний и максимальный ток — иначе модель спутает сдвиг по току со старением
  (§17.3: у Oxford заряд идёт на C/18.5, у CALCE и Sandia — на 0.5C);
* средняя температура окна;
* интеграл ``q_norm``.

Признаки считаются по равномерной сетке напряжений, которую построил генератор
фрагментов (§7), поэтому ``dQ/dV`` — это просто производная ``q_norm`` по
сетке, делённая на шаг.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import polars as pl
from scipy.signal import savgol_filter

from bsb.config import BenchmarkConfig

__all__ = ["FEATURE_PREFIX", "window_features"]

#: Префикс всех признаков в таблице.
FEATURE_PREFIX = "f_"


def _savgol(values: np.ndarray, window: int, polyorder: int) -> np.ndarray:
    """Сгладить ряд Савицким-Голеем, если длины хватает."""
    if values.size <= window:
        return values
    return savgol_filter(values, window_length=window, polyorder=polyorder)


def _window_row(
    q_norm: Sequence[float],
    current: Sequence[float],
    temperature: Sequence[float],
    *,
    v_start: float,
    v_end: float,
    savgol_window: int,
    savgol_polyorder: int,
) -> dict[str, float]:
    """Признаки одного окна."""
    charge = np.asarray(q_norm, dtype=float)
    current_array = np.asarray(current, dtype=float)
    temperature_array = np.asarray(temperature, dtype=float)
    grid = np.linspace(v_start, v_end, charge.size)
    step = (v_end - v_start) / max(charge.size - 1, 1)

    smooth = _savgol(charge, savgol_window, savgol_polyorder)
    dqdv = np.gradient(smooth, grid)
    half = charge.size // 2

    return {
        "v_start": v_start,
        "v_end": v_end,
        "v_width": v_end - v_start,
        "q_mean": float(charge.mean()),
        "q_std": float(charge.std()),
        "q_min": float(charge.min()),
        "q_max": float(charge.max()),
        "q_last": float(charge[-1]),
        "q_integral": float(np.trapezoid(charge, grid)),
        "q_slope": float((charge[-1] - charge[0]) / max(v_end - v_start, 1e-9)),
        "dqdv_mean": float(dqdv.mean()),
        "dqdv_std": float(dqdv.std()),
        "dqdv_min": float(dqdv.min()),
        "dqdv_max": float(dqdv.max()),
        "dqdv_mid": float(dqdv[half]),
        "dqdv_step": step,
        "current_mean": float(current_array.mean()),
        "current_abs_max": float(np.abs(current_array).max()),
        "current_std": float(current_array.std()),
        "temperature_mean": float(temperature_array.mean()),
        "temperature_std": float(temperature_array.std()),
        "temperature_min": float(temperature_array.min()),
        "temperature_max": float(temperature_array.max()),
    }


def window_features(fragments: pl.DataFrame, config: BenchmarkConfig) -> pl.DataFrame:
    """Построить таблицу признаков по фрагментам (ТЗ §12.5).

    Возвращает ``fragment_id``, признаки с префиксом ``f_`` и, если он был,
    столбец ``soh`` — цель остаётся рядом, чтобы удобно было делить на выборки.
    """
    if fragments.height == 0:
        return pl.DataFrame(schema={"fragment_id": pl.String})
    savgol = config.baseline
    rows: list[dict[str, object]] = []
    for row in fragments.iter_rows(named=True):
        features = _window_row(
            row["q_norm"],
            row["current"],
            row["temperature"],
            v_start=float(row["v_start"]),
            v_end=float(row["v_end"]),
            savgol_window=savgol.savgol_window_length,
            savgol_polyorder=savgol.savgol_polyorder,
        )
        rows.append(
            {
                "fragment_id": row["fragment_id"],
                "soh": row.get("soh"),
                **{f"{FEATURE_PREFIX}{key}": value for key, value in features.items()},
            }
        )
    return pl.DataFrame(rows)
