"""Синтетическая ячейка с известной ёмкостью.

Используется round-trip тестом §14 («синтетическая ячейка проходит
ingest -> labels -> fragments и восстанавливается») и любым тестом, которому
нужен валидный :class:`~bsb.ingest.base.RawBundle` или
:class:`~bsb.schema.CellBundle` без реальных данных.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from bsb.config import load_config
from bsb.ingest.base import RawBundle, build_raw_bundle, integrate_charge
from bsb.labels import build_labels
from bsb.schema import CellBundle

__all__ = ["synth_bundle", "synth_cycle_timeseries", "synth_raw_bundle"]


def synth_cycle_timeseries(
    cell_id: str,
    cycle_index: int,
    *,
    charge_current_a: float = 0.36,
    dt_s: float = 10.0,
    n_cc: int = 600,
) -> pl.DataFrame:
    """Один полный цикл: CC-заряд, CV-заряд, rest, CC-разряд.

    ``n_cc`` выбран так, чтобы в окно шириной 0.2 В попадало ~100 точек: иначе
    тесты упираются в ``fragment.min_points_in_window``, а не в логику.

    Все соглашения §4.3 соблюдены: ток заряда положительный, разряда
    отрицательный, ``charge_ah`` получен интегрированием внутри шага,
    ``t_s`` отсчитывается от начала цикла.
    """
    frames: list[pl.DataFrame] = []
    cursor = 0.0

    def add(kind: str, current: np.ndarray, voltage: np.ndarray) -> None:
        nonlocal cursor
        size = current.size
        t_local = np.arange(size, dtype=float) * dt_s
        frames.append(
            pl.DataFrame(
                {
                    "cell_id": [cell_id] * size,
                    "cycle_index": [cycle_index] * size,
                    "step_type": [kind] * size,
                    "t_s": cursor + t_local,
                    "voltage_v": voltage,
                    "current_a": current,
                    "temperature_c": np.full(size, 30.0),
                    "charge_ah": integrate_charge(t_local, current),
                }
            )
        )
        cursor += t_local[-1] + dt_s

    add("chg_cc", np.full(n_cc, charge_current_a), np.linspace(3.0, 4.2, n_cc))
    add("chg_cv", np.linspace(charge_current_a, 0.05 * charge_current_a, 20), np.full(20, 4.2))
    add("rest", np.zeros(5), np.full(5, 4.19))
    add("dch", np.full(n_cc, -charge_current_a), np.linspace(4.2, 3.0, n_cc))
    return pl.concat(frames)


def synth_raw_bundle(
    cell_id: str = "oxford_Cell1",
    *,
    n_cycles: int = 6,
    bol_capacity_ah: float = 0.72,
    fade_per_cycle: float = 0.005,
    dataset: str = "oxford",
    chemistry: str = "LCO",
    form_factor: str = "pouch",
    nominal_capacity_ah: float = 0.74,
    temperature_setpoint_c: float = 40.0,
    non_reference_cycles: tuple[int, ...] = (),
) -> RawBundle:
    """Ячейка до разметки: линейно деградирующая ёмкость и полный timeseries.

    ``non_reference_cycles`` помечает перечисленные (1-based) циклы как
    неопорные — нужно тестам интерполяции и отбрасывания.
    """
    capacities = [bol_capacity_ah * (1.0 - fade_per_cycle * index) for index in range(n_cycles)]
    cycles = pl.DataFrame(
        {
            "cell_id": [cell_id] * n_cycles,
            "cycle_index": np.arange(1, n_cycles + 1, dtype=np.int32),
            "capacity_ah": capacities,
            "is_reference": [index not in non_reference_cycles for index in range(1, n_cycles + 1)],
        }
    )
    timeseries = pl.concat(
        [synth_cycle_timeseries(cell_id, index + 1) for index in range(n_cycles)]
    )
    cell = {
        "cell_id": cell_id,
        "dataset": dataset,
        "chemistry": chemistry,
        "form_factor": form_factor,
        "nominal_capacity_ah": nominal_capacity_ah,
        "temperature_setpoint_c": temperature_setpoint_c,
        "charge_protocol": "CC-CV 0.5C to 4.2V, cutoff 0.05A",
        "batch_id": None,
        "notes": None,
    }
    return build_raw_bundle(cell, cycles, timeseries)


def synth_bundle(**kwargs: object) -> CellBundle:
    """Размеченный бандл по схеме §4: ``synth_raw_bundle`` + ``build_labels``."""
    return build_labels(synth_raw_bundle(**kwargs), load_config())  # type: ignore[arg-type]
