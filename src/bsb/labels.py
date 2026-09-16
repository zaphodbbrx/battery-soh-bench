"""Опорные циклы и целевая переменная (ТЗ §5).

Разделение ответственности, зафиксированное в ТЗ §5:

* **какие циклы считать опорными** — решение владельца репозитория, по одному
  на датасет (``labels.reference_cycle_rules`` в конфиге); применяется в
  ``DatasetAdapter.mark_reference_cycles`` и принимается глазами по графикам
  §14, а не юнит-тестом;
* **арифметика поверх опорных циклов** — ``soh = capacity_ah / bol``,
  интерполяция, ``soh_gap_cycles``, запрет экстраполяции — реализована здесь
  и не содержит датасет-специфичных ветвлений.

Инварианты §5, обеспечиваемые механически:

1. ``bol_capacity_ah`` — ёмкость **этой** ячейки на первом опорном цикле;
2. экстраполяция за пределы последнего опорного цикла запрещена — такие
   циклы отбрасываются, а не дотягиваются;
3. ``soh_source = interpolated`` ставится только там, где оба соседних
   опорных измерения существуют.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from bsb.config import BenchmarkConfig
from bsb.ingest.base import RawBundle
from bsb.schema import (
    CELLS_SCHEMA,
    CYCLES_SCHEMA,
    SOH_MAX,
    CellBundle,
    conform,
)

__all__ = [
    "LabelsError",
    "build_labels",
    "gap_to_reference",
    "interpolate_reference_soh",
]


class LabelsError(ValueError):
    """Разметка невозможна: нет опорных измерений или ёмкость не измерена."""


def gap_to_reference(cycle_index: np.ndarray, reference_index: np.ndarray) -> np.ndarray:
    """Расстояние от каждого цикла до ближайшего опорного (в циклах).

    Parameters
    ----------
    cycle_index
        Индексы всех циклов ячейки.
    reference_index
        Индексы опорных циклов (непустой, возрастающий).
    """
    if reference_index.size == 0:
        raise LabelsError("gap_to_reference: пустой список опорных циклов")
    distances = np.abs(cycle_index[:, None] - reference_index[None, :])
    return distances.min(axis=1).astype(np.int32)


def interpolate_reference_soh(
    cycle_index: np.ndarray,
    reference_index: np.ndarray,
    reference_soh: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Линейная интерполяция SOH между опорными циклами.

    Экстраполяция запрещена (ТЗ §5 п.5): для циклов вне отрезка
    ``[первый опорный, последний опорный]`` значение не определено.

    Returns
    -------
    tuple of numpy.ndarray
        ``(values, defined)`` — интерполированные значения и маска, где они
        определены (внутри отрезка опорных измерений).
    """
    if reference_index.size < 2:
        raise LabelsError(
            "interpolate_reference_soh: нужно минимум два опорных цикла, "
            f"получено {reference_index.size}"
        )
    values = np.interp(cycle_index, reference_index, reference_soh)
    defined = (cycle_index >= reference_index[0]) & (cycle_index <= reference_index[-1])
    return values, defined


def build_labels(raw: RawBundle, config: BenchmarkConfig) -> CellBundle:
    """Посчитать ``bol_capacity_ah``, ``soh``, ``soh_source`` и ``soh_gap_cycles``.

    Parameters
    ----------
    raw
        Ячейка с размеченными опорными циклами (``RawBundle.cycles.is_reference``).
    config
        Конфиг: ``labels.reference_only``, ``labels.allow_interpolated``,
        ``labels.max_soh_gap_cycles``, ``labels.bol_source``.

    Returns
    -------
    CellBundle
        Кадры по схеме §4, готовые к записи в хранилище.

    Raises
    ------
    LabelsError
        Если опорных измерений нет, ``bol`` не определена или выбранный
        ``bol_source``/режим ещё не поддержан.
    """
    labels = config.labels
    if labels.bol_source != "first_reference_cycle":
        raise LabelsError(
            f"labels.bol_source={labels.bol_source!r} пока не поддержан: "
            "реализован только first_reference_cycle (ТЗ §5 п.1)"
        )
    cycles = raw.cycles
    if cycles.height == 0:
        raise LabelsError(f"{raw.cell_id}: нет циклов")
    duplicate = cycles.select(["cell_id", "cycle_index"]).is_duplicated()
    if duplicate.any():
        raise LabelsError(f"{raw.cell_id}: дубли (cell_id, cycle_index)")

    measured = cycles.filter(pl.col("is_reference") & pl.col("capacity_ah").is_not_null()).sort(
        "cycle_index"
    )
    if measured.height == 0:
        raise LabelsError(
            f"{raw.cell_id}: нет ни одного опорного цикла с измеренной ёмкостью "
            "(ТЗ §5 п.1: bol взять негде)"
        )
    bol = float(measured["capacity_ah"][0])
    if not np.isfinite(bol) or bol <= 0:
        raise LabelsError(f"{raw.cell_id}: bol_capacity_ah={bol} не положительна")

    reference_index = measured["cycle_index"].to_numpy()
    reference_soh = measured["capacity_ah"].to_numpy() / bol

    if labels.reference_only or not labels.allow_interpolated:
        # §5.1: фрагменты строятся только из опорных измерений, интерполяция
        # не нужна вовсе — soh всегда measured, разрыв всегда 0.
        labelled = measured.with_columns(
            (pl.col("capacity_ah") / bol).alias("soh"),
            pl.lit("measured").cast(CYCLES_SCHEMA["soh_source"]).alias("soh_source"),
            pl.lit(0, dtype=pl.Int32).alias("soh_gap_cycles"),
        )
    else:
        labelled = _label_with_interpolation(
            cycles, reference_index, reference_soh, bol, max_gap=labels.max_soh_gap_cycles
        )

    if labelled.height == 0:
        raise LabelsError(f"{raw.cell_id}: после применения правил §5 не осталось ни одного цикла")
    bad = labelled.filter((pl.col("soh") <= 0) | (pl.col("soh") > SOH_MAX))
    if bad.height:
        worst = bad.sort("soh", descending=True).head(3).select("cycle_index", "soh").to_dicts()
        raise LabelsError(
            f"{raw.cell_id}: {bad.height} циклов получили soh вне (0, {SOH_MAX}]; примеры: {worst}"
        )

    cells = raw.cells.with_columns(pl.lit(bol).alias("bol_capacity_ah"))
    return CellBundle(
        cells=conform(cells, CELLS_SCHEMA, name="cells"),
        cycles=conform(labelled.sort("cycle_index"), CYCLES_SCHEMA, name="cycles"),
        timeseries=raw.timeseries,
    )


def _label_with_interpolation(
    cycles: pl.DataFrame,
    reference_index: np.ndarray,
    reference_soh: np.ndarray,
    bol: float,
    *,
    max_gap: int,
) -> pl.DataFrame:
    """Разметить циклы в режиме интерполяции (§5 п.3-4).

    Опорные измерения остаются ``measured``; остальные получают линейно
    интерполированное значение, если лежат внутри отрезка опорных измерений и
    не дальше ``max_gap`` циклов от ближайшего опорного.
    """
    all_index = cycles["cycle_index"].to_numpy()
    is_reference = cycles["is_reference"].to_numpy()
    has_capacity = cycles["capacity_ah"].is_not_null().to_numpy()
    measured_mask = is_reference & has_capacity

    interpolated, defined = interpolate_reference_soh(all_index, reference_index, reference_soh)
    gaps = gap_to_reference(all_index, reference_index)
    keep = measured_mask | (defined & (gaps <= max_gap))
    soh = np.where(
        measured_mask, np.asarray(cycles["capacity_ah"].to_numpy(), dtype=float) / bol, interpolated
    )
    source = np.where(measured_mask, "measured", "interpolated")
    return (
        cycles.with_columns(
            pl.Series("soh", soh, dtype=pl.Float64),
            pl.Series("soh_source", source, dtype=pl.String).cast(CYCLES_SCHEMA["soh_source"]),
            pl.Series("soh_gap_cycles", gaps, dtype=pl.Int32),
            pl.Series("_keep", keep, dtype=pl.Boolean),
        )
        .filter(pl.col("_keep"))
        .drop("_keep")
    )
