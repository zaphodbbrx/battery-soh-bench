"""Общий контракт адаптеров датасетов (ТЗ §6).

Весь датасет-специфичный беспорядок живёт внутри адаптера и наружу не
протекает: на выходе всегда :class:`bsb.schema.CellBundle` по схеме §4.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
import polars as pl

from bsb.schema import CELLS_SCHEMA, CYCLES_SCHEMA, TIMESERIES_SCHEMA, CellBundle, conform

__all__ = [
    "CellSource",
    "DatasetAdapter",
    "build_bundle",
    "cell_id_for",
    "integrate_charge",
]


def cell_id_for(dataset: str, original_id: str) -> str:
    """Собрать глобально уникальный ``cell_id`` вида ``{dataset}_{original_id}``."""
    return f"{dataset}_{original_id}"


@dataclass(frozen=True, slots=True)
class CellSource:
    """Одна ячейка, найденная в сырых данных.

    ``extras`` — свободный словарь для датасет-специфичных подробностей
    (номер партии, путь к листу Excel, ссылка на строку в таблице check-up).
    Наружу за пределы адаптера он не выходит.
    """

    dataset: str
    original_id: str
    path: Path
    extras: Mapping[str, Any] = field(default_factory=dict)

    @property
    def cell_id(self) -> str:
        """Глобально уникальный идентификатор ячейки."""
        return cell_id_for(self.dataset, self.original_id)


@runtime_checkable
class DatasetAdapter(Protocol):
    """Контракт адаптера (ТЗ §6)."""

    name: str

    def discover(self, raw_dir: Path) -> list[CellSource]:
        """Найти ячейки в каталоге сырых данных."""
        ...

    def load_cell(self, src: CellSource) -> CellBundle:
        """Прочитать одну ячейку и вернуть бандл по схеме §4."""
        ...

    def mark_reference_cycles(self, bundle: CellBundle) -> CellBundle:
        """Разметить опорные циклы согласно ``labels.reference_cycle_rules``."""
        ...


def integrate_charge(t_s: Sequence[float], current_a: Sequence[float]) -> np.ndarray:
    """Накопленный заряд от начала шага, А·ч (ТЗ §4.3).

    Считается трапециями по времени и току, а не берётся из исходника: в
    исходных файлах соглашения о сбросе счётчика разные. Первая точка — 0.
    """
    time = np.asarray(t_s, dtype=float)
    current = np.asarray(current_a, dtype=float)
    if time.shape != current.shape:
        raise ValueError(f"integrate_charge: формы не совпадают: {time.shape} и {current.shape}")
    if time.size == 0:
        return np.zeros(0, dtype=float)
    deltas = np.diff(time) / 3600.0
    trapezoids = 0.5 * (current[1:] + current[:-1]) * deltas
    return np.concatenate([[0.0], np.cumsum(trapezoids)])


def build_bundle(
    cell: Mapping[str, Any],
    cycles: pl.DataFrame,
    timeseries: pl.DataFrame,
) -> CellBundle:
    """Собрать бандл из «как получилось» кадров адаптера и привести к §4."""
    cells = pl.DataFrame([dict(cell)], schema=dict(CELLS_SCHEMA))
    return CellBundle(
        cells=conform(cells, CELLS_SCHEMA, name="cells"),
        cycles=conform(cycles, CYCLES_SCHEMA, name="cycles"),
        timeseries=conform(timeseries, TIMESERIES_SCHEMA, name="timeseries"),
    )
