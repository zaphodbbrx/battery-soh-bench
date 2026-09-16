"""Тесты разметки: арифметика §5 поверх опорных циклов."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import polars as pl
import pytest

from bsb.config import BenchmarkConfig
from bsb.ingest.base import RawBundle
from bsb.labels import LabelsError, build_labels, gap_to_reference, interpolate_reference_soh
from synthetic import synth_raw_bundle


def test_bol_is_taken_from_this_cell(synthetic_bundle: Any) -> None:
    """bol — ёмкость этой ячейки на первом опорном цикле, а не паспортная (§5 п.1)."""
    cell = synthetic_bundle.cell
    assert cell["bol_capacity_ah"] == pytest.approx(0.72)
    assert cell["bol_capacity_ah"] != cell["nominal_capacity_ah"]


def test_reference_only_mode_marks_everything_measured(repo_config: BenchmarkConfig) -> None:
    """§5.1: в режиме «только опорные» soh всегда measured, разрыв всегда 0."""
    raw = synth_raw_bundle(n_cycles=5, non_reference_cycles=(3,))
    bundle = build_labels(raw, repo_config)
    assert bundle.cycles.height == 4, "неопорный цикл не должен попадать в бенчмарк"
    assert bundle.cycles["soh_source"].unique().to_list() == ["measured"]
    assert bundle.cycles["soh_gap_cycles"].unique().to_list() == [0]
    assert bundle.cycles["is_reference"].all()


def test_interpolated_mode_fills_gaps_and_marks_source(
    make_config: Callable[[dict[str, Any] | None], BenchmarkConfig],
) -> None:
    """§5 п.3: неопорные циклы получают линейно интерполированный soh."""
    config = make_config({"labels.reference_only": False, "labels.allow_interpolated": True})
    raw = synth_raw_bundle(n_cycles=5, non_reference_cycles=(3,))
    bundle = build_labels(raw, config)
    assert bundle.cycles.height == 5
    row = bundle.cycles.filter(pl.col("cycle_index") == 3).row(0, named=True)
    assert row["soh_source"] == "interpolated"
    assert row["is_reference"] is False
    assert row["soh_gap_cycles"] == 1
    # линейная интерполяция между 2-м и 4-м циклами (шаг деградации постоянный)
    neighbours = bundle.cycles.filter(pl.col("cycle_index").is_in([2, 4]))["soh"].to_list()
    assert row["soh"] == pytest.approx(sum(neighbours) / 2)


def test_interpolation_refuses_extrapolation(
    make_config: Callable[[dict[str, Any] | None], BenchmarkConfig],
) -> None:
    """§5 п.5: за последним опорным циклом значение не дотягивается."""
    config = make_config({"labels.reference_only": False, "labels.allow_interpolated": True})
    # опорные — только циклы 2 и 3 из пяти
    raw = synth_raw_bundle(n_cycles=5, non_reference_cycles=(1, 4, 5))
    bundle = build_labels(raw, config)
    kept = bundle.cycles["cycle_index"].to_list()
    assert kept == [2, 3], "циклы вне отрезка опорных измерений должны быть отброшены"


def test_max_gap_filters_far_cycles(
    make_config: Callable[[dict[str, Any] | None], BenchmarkConfig],
) -> None:
    """§5 п.4: слишком далёкий от опорного измерения цикл отбрасывается."""
    strict = make_config(
        {
            "labels.reference_only": False,
            "labels.allow_interpolated": True,
            "labels.max_soh_gap_cycles": 1,
        }
    )
    raw = synth_raw_bundle(n_cycles=7, non_reference_cycles=(4, 5, 6))
    bundle = build_labels(raw, strict)
    # опорные 1, 2, 3, 7; при max_soh_gap_cycles=1 циклы 4 и 6 (разрыв 1)
    # остаются, а цикл 5 (разрыв 2) отбрасывается
    assert bundle.cycles["cycle_index"].to_list() == [1, 2, 3, 4, 6, 7]
    assert bundle.cycles["soh_gap_cycles"].max() <= 1


def test_no_reference_measurement_is_an_error(repo_config: BenchmarkConfig) -> None:
    """Без опорных измерений bol взять негде — внятная ошибка."""
    raw = synth_raw_bundle(n_cycles=3)
    broken = RawBundle(
        cells=raw.cells,
        cycles=raw.cycles.with_columns(pl.lit(False).alias("is_reference")),
        timeseries=raw.timeseries,
    )
    with pytest.raises(LabelsError, match="опорного цикла"):
        build_labels(broken, repo_config)


def test_capacity_out_of_range_is_rejected(repo_config: BenchmarkConfig) -> None:
    """Ёмкость, дающая soh > 1.2, — ошибка разметки, а не молчаливый пропуск."""
    raw = synth_raw_bundle(n_cycles=3)
    cells = raw.cycles.with_columns(
        pl.when(pl.col("cycle_index") == 3)
        .then(pl.col("capacity_ah") * 1.5)
        .otherwise(pl.col("capacity_ah"))
        .alias("capacity_ah")
    )
    broken = RawBundle(cells=raw.cells, cycles=cells, timeseries=raw.timeseries)
    with pytest.raises(LabelsError, match="soh вне"):
        build_labels(broken, repo_config)


def test_gap_to_reference_uses_nearest() -> None:
    """Разрыв считается до ближайшего опорного цикла."""
    gaps = gap_to_reference(np.array([1, 5, 9, 15]), np.array([1, 10]))
    assert gaps.tolist() == [0, 4, 1, 5]


def test_interpolate_marks_only_inside_range() -> None:
    """Вне отрезка опорных измерений значение помечено как неопределённое."""
    values, defined = interpolate_reference_soh(
        np.array([1, 2, 3, 4, 5]), np.array([2, 4]), np.array([0.9, 0.7])
    )
    assert defined.tolist() == [False, True, True, True, False]
    assert values[2] == pytest.approx(0.8)


def test_single_reference_cannot_interpolate() -> None:
    """Один опорный цикл интерполировать нечем."""
    with pytest.raises(LabelsError, match="минимум два"):
        interpolate_reference_soh(np.array([1, 2]), np.array([1]), np.array([0.9]))
