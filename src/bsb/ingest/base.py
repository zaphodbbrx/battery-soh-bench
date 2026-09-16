"""Общий контракт адаптеров датасетов (ТЗ §6).

Весь датасет-специфичный беспорядок живёт внутри адаптера и наружу не
протекает: на выходе всегда :class:`bsb.schema.CellBundle` по схеме §4.

Двухступенчатая схема (уточнение ТЗ §6)
---------------------------------------
ТЗ §6 предполагает, что ``load_cell`` сразу возвращает ``CellBundle`` по схеме
§4. Это невозможно без подлога: поля ``cells.bol_capacity_ah`` (§4.1) и
``cycles.soh`` / ``soh_source`` / ``soh_gap_cycles`` (§4.2) **выводятся** из
опорных циклов, а какие циклы опорные — решает правило датасета, которое
применяется на следующем шаге. Поэтому адаптер возвращает промежуточный
:class:`RawBundle` (ё̈мкость и timeseries как они есть в исходнике), а
``mark_reference_cycles`` размечает опорные циклы и через
:func:`bsb.labels.build_labels` отдаёт полноценный :class:`~bsb.schema.CellBundle`.

Так ни одно производное поле не заполняется «чем-нибудь похожим».
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Protocol, runtime_checkable

import numpy as np
import polars as pl

from bsb.schema import (
    CELLS_SCHEMA,
    TIMESERIES_SCHEMA,
    CellBundle,
    SchemaViolation,
    check_structure,
    conform,
    validate_timeseries,
)

__all__ = [
    "RAW_CELLS_SCHEMA",
    "RAW_CYCLES_SCHEMA",
    "CellSource",
    "DatasetAdapter",
    "RawBundle",
    "build_raw_bundle",
    "cell_id_for",
    "integrate_charge",
    "validate_raw_bundle",
]

#: ``cells`` до разметки: всё из §4.1, кроме выводимой ``bol_capacity_ah``.
RAW_CELLS_SCHEMA: Final[Mapping[str, pl.DataType]] = {
    key: dtype for key, dtype in CELLS_SCHEMA.items() if key != "bol_capacity_ah"
}

#: ``cycles`` до разметки: только измеренное и признак опорного цикла.
RAW_CYCLES_SCHEMA: Final[Mapping[str, pl.DataType]] = {
    "cell_id": pl.String,
    "cycle_index": pl.Int32,
    "capacity_ah": pl.Float64,
    "is_reference": pl.Boolean,
}


def cell_id_for(dataset: str, original_id: str) -> str:
    """Собрать глобально уникальный ``cell_id`` вида ``{dataset}_{original_id}``."""
    return f"{dataset}_{original_id}"


@dataclass(frozen=True, slots=True)
class CellSource:
    """Одна ячейка, найденная в сырых данных.

    ``extras`` — свободный словарь для датасет-специфичных подробностей
    (имя поля в ``.mat``, путь к листу Excel, ссылка на строку в таблице
    check-up). Наружу за пределы адаптера он не выходит.
    """

    dataset: str
    original_id: str
    path: Path
    extras: Mapping[str, Any] = field(default_factory=dict)

    @property
    def cell_id(self) -> str:
        """Глобально уникальный идентификатор ячейки."""
        return cell_id_for(self.dataset, self.original_id)


@dataclass(frozen=True, slots=True)
class RawBundle:
    """Ячейка до применения правила опорных циклов (см. модульный docstring)."""

    cells: pl.DataFrame
    cycles: pl.DataFrame
    timeseries: pl.DataFrame

    @property
    def cell(self) -> dict[str, Any]:
        """Единственная строка ``cells`` как словарь."""
        if self.cells.height != 1:
            raise SchemaViolation(
                f"RawBundle.cells должен содержать ровно одну строку, получено {self.cells.height}"
            )
        return self.cells.row(0, named=True)

    @property
    def cell_id(self) -> str:
        """Идентификатор ячейки."""
        return str(self.cell["cell_id"])


@runtime_checkable
class DatasetAdapter(Protocol):
    """Контракт адаптера (ТЗ §6 с уточнением про ``RawBundle``)."""

    name: str

    def discover(self, raw_dir: Path) -> list[CellSource]:
        """Найти ячейки в каталоге сырых данных."""
        ...

    def load_cell(self, src: CellSource) -> RawBundle:
        """Прочитать одну ячейку: ёмкости по циклам и timeseries по §4.3."""
        ...

    def mark_reference_cycles(self, bundle: RawBundle) -> CellBundle:
        """Разметить опорные циклы по правилу датасета и посчитать §4.2."""
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


def build_raw_bundle(
    cell: Mapping[str, Any],
    cycles: pl.DataFrame,
    timeseries: pl.DataFrame,
) -> RawBundle:
    """Собрать ``RawBundle`` из «как получилось» кадров адаптера."""
    cells = pl.DataFrame([dict(cell)], schema=dict(RAW_CELLS_SCHEMA))
    return RawBundle(
        cells=conform(cells, RAW_CELLS_SCHEMA, name="cells (raw)"),
        cycles=conform(cycles, RAW_CYCLES_SCHEMA, name="cycles (raw)"),
        timeseries=conform(timeseries, TIMESERIES_SCHEMA, name="timeseries"),
    )


def validate_raw_bundle(bundle: RawBundle, *, strict: bool = True) -> list[str]:
    """Проверить бандл до разметки: схемы, уникальность циклов, соглашения §4.3."""
    check_structure(bundle.cells, RAW_CELLS_SCHEMA, name="cells (raw)")
    check_structure(bundle.cycles, RAW_CYCLES_SCHEMA, name="cycles (raw)")
    if bundle.cells.height != 1:
        raise SchemaViolation(
            f"RawBundle.cells: ожидалась одна строка, получено {bundle.cells.height}"
        )
    cycles = bundle.cycles
    if cycles.height == 0:
        raise SchemaViolation(f"RawBundle {bundle.cell_id}: нет ни одного цикла")
    duplicates = cycles.select(["cell_id", "cycle_index"]).is_duplicated()
    if duplicates.any():
        raise SchemaViolation(f"RawBundle {bundle.cell_id}: дубли (cell_id, cycle_index)")
    bad_index = cycles["cycle_index"] < 1
    if bad_index.any():
        raise SchemaViolation(f"RawBundle {bundle.cell_id}: cycle_index должен быть >= 1")
    bad_capacity = cycles["capacity_ah"].is_not_null() & (
        ~cycles["capacity_ah"].is_finite() | (cycles["capacity_ah"] <= 0)
    )
    if bad_capacity.any():
        raise SchemaViolation(
            f"RawBundle {bundle.cell_id}: capacity_ah либо null, либо > 0 "
            f"(нули вместо пропусков запрещены)"
        )
    foreign = set(cycles["cell_id"].to_list()) - {bundle.cell_id}
    if foreign:
        raise SchemaViolation(
            f"RawBundle {bundle.cell_id}: чужие cell_id в cycles: {sorted(foreign)}"
        )
    series_cycles = set(bundle.timeseries["cycle_index"].to_list())
    missing = series_cycles - set(cycles["cycle_index"].to_list())
    if missing:
        raise SchemaViolation(
            f"RawBundle {bundle.cell_id}: циклы timeseries отсутствуют в cycles: "
            f"{sorted(missing)[:10]}"
        )
    return validate_timeseries(bundle.timeseries, strict=strict)
