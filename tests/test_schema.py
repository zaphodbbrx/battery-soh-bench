"""Тесты схемы §4 и валидаторов."""

from __future__ import annotations

import io
from collections.abc import Callable

import polars as pl
import pytest

from bsb.ingest.base import integrate_charge
from bsb.schema import (
    CELLS_SCHEMA,
    CYCLES_SCHEMA,
    FRAGMENTS_SCHEMA,
    FRAGMENTS_UNLABELED_SCHEMA,
    TIMESERIES_SCHEMA,
    CellBundle,
    SchemaViolation,
    conform,
    empty_frame,
    validate_bundle,
    validate_cells,
    validate_cycles,
    validate_fragments,
    validate_timeseries,
)


def _valid_fragment_row(
    fragment_id: str = "f0", n_points: int = 4, **overrides: object
) -> dict[str, object]:
    """Строка фрагмента, которую валидатор обязан принять."""
    row: dict[str, object] = {
        "fragment_id": fragment_id,
        "cell_id": "oxford_Cell1",
        "cycle_index": 1,
        "v_start": 3.6,
        "v_end": 3.8,
        "q_norm": [0.0, 0.1, 0.2, 0.3][:n_points],
        "current": [0.36] * n_points,
        "temperature": [30.0] * n_points,
        "soh": 0.95,
    }
    row.update(overrides)
    return row


# --------------------------------------------------------------------------
# Структура
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("schema", "validator"),
    [
        (CELLS_SCHEMA, validate_cells),
        (CYCLES_SCHEMA, validate_cycles),
        (FRAGMENTS_SCHEMA, validate_fragments),
    ],
)
def test_empty_frames_are_valid(
    schema: dict[str, pl.DataType], validator: Callable[[pl.DataFrame], list[str]]
) -> None:
    """Пустой кадр правильной схемы — не ошибка."""
    assert validator(empty_frame(schema)) == []


def test_empty_timeseries_requires_nonempty_flag() -> None:
    """Пустой timeseries — ошибка, если не отключить требование непустоты."""
    with pytest.raises(SchemaViolation, match="пустой кадр"):
        validate_timeseries(empty_frame(TIMESERIES_SCHEMA))
    assert validate_timeseries(empty_frame(TIMESERIES_SCHEMA), require_nonempty=False) == []


def test_conform_casts_strings_to_enum_and_orders_columns() -> None:
    """``conform`` приводит строки к Enum и расставляет колонки по схеме."""
    loose = pl.DataFrame(
        {
            "dataset": ["oxford"],
            "cell_id": ["oxford_Cell1"],
            "chemistry": ["LCO"],
            "form_factor": ["pouch"],
            "nominal_capacity_ah": [0.74],
            "bol_capacity_ah": [0.72],
            "temperature_setpoint_c": [40.0],
            "charge_protocol": ["CC-CV"],
            "batch_id": [None],
            "notes": [None],
        }
    )
    result = conform(loose, CELLS_SCHEMA, name="cells")
    assert list(result.columns) == list(CELLS_SCHEMA)
    assert result.schema["dataset"] == CELLS_SCHEMA["dataset"]
    validate_cells(result)


def test_conform_rejects_unknown_category() -> None:
    """Постороннее значение controlled vocabulary не проходит."""
    loose = pl.DataFrame(
        {
            "cell_id": ["tesla_Cell1"],
            "dataset": ["tesla"],
            "chemistry": ["LCO"],
            "form_factor": ["pouch"],
            "nominal_capacity_ah": [0.74],
            "bol_capacity_ah": [0.72],
            "temperature_setpoint_c": [40.0],
            "charge_protocol": ["CC-CV"],
            "batch_id": [None],
            "notes": [None],
        }
    )
    with pytest.raises(SchemaViolation, match="привести типы"):
        conform(loose, CELLS_SCHEMA, name="cells")


def test_conform_reports_missing_and_extra_columns() -> None:
    """Отсутствующие и лишние колонки — разные внятные ошибки."""
    with pytest.raises(SchemaViolation, match="отсутствуют колонки"):
        conform(pl.DataFrame({"cell_id": ["a"]}), CELLS_SCHEMA, name="cells")
    with pytest.raises(SchemaViolation, match="лишние колонки"):
        conform(
            empty_frame(CELLS_SCHEMA).with_columns(pl.lit(1).alias("junk")),
            CELLS_SCHEMA,
            name="cells",
        )


def test_enum_survives_parquet_roundtrip() -> None:
    """Enum не деградирует до строки при записи в parquet (ТЗ §4)."""
    frame = empty_frame(CELLS_SCHEMA)
    buffer = io.BytesIO()
    frame.write_parquet(buffer)
    buffer.seek(0)
    restored = pl.read_parquet(buffer)
    assert restored.schema["dataset"] == CELLS_SCHEMA["dataset"]
    validate_cells(restored)


def test_structure_check_rejects_wrong_order() -> None:
    """Порядок колонок — часть контракта."""
    frame = empty_frame(CELLS_SCHEMA).select(sorted(CELLS_SCHEMA))
    with pytest.raises(SchemaViolation, match="порядок колонок"):
        validate_cells(frame)


# --------------------------------------------------------------------------
# cells
# --------------------------------------------------------------------------


def test_validate_cells_accepts_synthetic(synthetic_bundle: CellBundle) -> None:
    """Синтетическая ячейка валидна целиком."""
    assert validate_bundle(synthetic_bundle) == []


def test_validate_cells_catches_duplicate_and_bad_prefix(synthetic_bundle: CellBundle) -> None:
    """Дубли cell_id и нарушение префикса ``{dataset}_`` ловятся."""
    doubled = pl.concat([synthetic_bundle.cells, synthetic_bundle.cells], how="vertical")
    with pytest.raises(SchemaViolation, match="не уникален"):
        validate_cells(doubled)
    mismatched = synthetic_bundle.cells.with_columns(pl.lit("calce_Cell1").alias("cell_id"))
    with pytest.raises(SchemaViolation, match="вид"):
        validate_cells(mismatched)


def test_validate_cells_rejects_nonpositive_capacity(synthetic_bundle: CellBundle) -> None:
    """Нулевая или отрицательная ёмкость — ошибка, а не пропуск."""
    broken = synthetic_bundle.cells.with_columns(pl.lit(0.0).alias("bol_capacity_ah"))
    with pytest.raises(SchemaViolation, match="bol_capacity_ah"):
        validate_cells(broken)


# --------------------------------------------------------------------------
# cycles
# --------------------------------------------------------------------------


def test_validate_cycles_catches_duplicate_cycle(synthetic_bundle: CellBundle) -> None:
    """Дубль (cell_id, cycle_index) — ошибка."""
    doubled = pl.concat([synthetic_bundle.cycles, synthetic_bundle.cycles.head(1)], how="vertical")
    with pytest.raises(SchemaViolation, match="дубли"):
        validate_cycles(doubled)


def test_validate_cycles_rejects_out_of_range_soh(synthetic_bundle: CellBundle) -> None:
    """soh вне (0, 1.2] отвергается."""
    for bad in (0.0, 1.3):
        broken = synthetic_bundle.cycles.with_columns(pl.lit(bad).alias("soh"))
        with pytest.raises(SchemaViolation, match="soh"):
            validate_cycles(broken)


def test_validate_cycles_enforces_soh_source_invariants(synthetic_bundle: CellBundle) -> None:
    """measured требует опорного цикла и нулевого разрыва; interpolated — ненулевого."""
    measured_with_gap = synthetic_bundle.cycles.with_columns(
        pl.lit(7, dtype=pl.Int32).alias("soh_gap_cycles")
    )
    with pytest.raises(SchemaViolation, match="measured"):
        validate_cycles(measured_with_gap)
    interpolated_without_gap = synthetic_bundle.cycles.with_columns(
        pl.lit("interpolated").cast(pl.Enum(["measured", "interpolated"])).alias("soh_source")
    )
    with pytest.raises(SchemaViolation, match="interpolated"):
        validate_cycles(interpolated_without_gap)


def test_validate_cycles_rejects_zero_capacity_instead_of_null(
    synthetic_bundle: CellBundle,
) -> None:
    """Ноль вместо пропуска в capacity_ah запрещён (ТЗ §4.3)."""
    broken = synthetic_bundle.cycles.with_columns(pl.lit(0.0).alias("capacity_ah"))
    with pytest.raises(SchemaViolation, match="capacity_ah"):
        validate_cycles(broken)


# --------------------------------------------------------------------------
# timeseries
# --------------------------------------------------------------------------


def test_integrate_charge_matches_manual_trapezoid() -> None:
    """``charge_ah`` считается интегрированием, а не берётся из исходника."""
    t_s = [0.0, 1.0, 2.0]
    current = [1.0, 2.0, 3.0]
    charge = integrate_charge(t_s, current)
    expected = [0.0, (1.0 + 2.0) / 2 / 3600, (1.0 + 2.0) / 2 / 3600 + (2.0 + 3.0) / 2 / 3600]
    assert charge.tolist() == pytest.approx(expected)


def test_validate_timeseries_accepts_synthetic(synthetic_bundle: CellBundle) -> None:
    """Синтетический timeseries соответствует §4.3 без замечаний."""
    assert validate_timeseries(synthetic_bundle.timeseries, strict=True) == []


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda df: df.with_columns((pl.col("t_s") * -1).alias("t_s")), "t_s"),
        (
            lambda df: df.with_columns(
                pl.when(pl.col("step_type") == pl.lit("dch"))
                .then(pl.lit(0.36))
                .otherwise(pl.col("current_a"))
                .alias("current_a")
            ),
            "dch",
        ),
        (
            lambda df: df.with_columns(
                pl.when(pl.col("step_type") == pl.lit("chg_cc"))
                .then(pl.lit(-0.36))
                .otherwise(pl.col("current_a"))
                .alias("current_a")
            ),
            "chg_cc",
        ),
        (lambda df: df.with_columns(pl.lit(0.5).alias("charge_ah")), "charge_ah"),
        (
            lambda df: df.with_columns(
                pl.when(pl.col("step_type") == pl.lit("rest"))
                .then(pl.lit(0.4))
                .otherwise(pl.col("current_a"))
                .alias("current_a")
            ),
            "rest",
        ),
    ],
)
def test_validate_timeseries_catches_convention_violations(
    synthetic_bundle: CellBundle,
    mutate: Callable[[pl.DataFrame], pl.DataFrame],
    message: str,
) -> None:
    """Каждое соглашение §4.3 ломается отдельным тестом."""
    broken = mutate(synthetic_bundle.timeseries)
    with pytest.raises(SchemaViolation, match=message):
        validate_timeseries(broken, strict=False)


def test_validate_timeseries_catches_charge_not_integrated(synthetic_bundle: CellBundle) -> None:
    """Расхождение charge_ah с интегралом тока — замечание, а не тишина."""
    broken = synthetic_bundle.timeseries.with_columns(
        (pl.col("charge_ah") * 1.5).alias("charge_ah")
    )
    warnings = validate_timeseries(broken, strict=True)
    assert any("интеграл" in warning for warning in warnings)


def test_validate_timeseries_catches_cycle_start_not_zero(synthetic_bundle: CellBundle) -> None:
    """Цикл обязан начинаться с t_s = 0 (ТЗ §4.3)."""
    broken = synthetic_bundle.timeseries.with_columns((pl.col("t_s") + 5.0).alias("t_s"))
    with pytest.raises(SchemaViolation, match="с нуля"):
        validate_timeseries(broken, strict=False)


# --------------------------------------------------------------------------
# fragments
# --------------------------------------------------------------------------


def test_validate_fragments_accepts_labeled_row() -> None:
    """Корректный labeled-фрагмент проходит."""
    frame = pl.DataFrame([_valid_fragment_row()], schema=dict(FRAGMENTS_SCHEMA))
    assert validate_fragments(frame, labeled=True, n_points=4) == []


def test_unlabeled_schema_hides_identity_and_target() -> None:
    """В тестовом файле участников нет cell_id, cycle_index и soh (ТЗ §7)."""
    assert "cell_id" not in FRAGMENTS_UNLABELED_SCHEMA
    assert "cycle_index" not in FRAGMENTS_UNLABELED_SCHEMA
    assert "soh" not in FRAGMENTS_UNLABELED_SCHEMA
    row = _valid_fragment_row()
    for key in ("cell_id", "cycle_index", "soh"):
        row.pop(key)
    frame = pl.DataFrame([row], schema=dict(FRAGMENTS_UNLABELED_SCHEMA))
    assert validate_fragments(frame, labeled=False, n_points=4) == []
    with pytest.raises(SchemaViolation, match="лишние колонки"):
        validate_fragments(
            pl.DataFrame([_valid_fragment_row()], schema=dict(FRAGMENTS_SCHEMA)),
            labeled=False,
            n_points=4,
        )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda row: row | {"q_norm": [0.0, 0.1]}, "длину 4"),
        (lambda row: row | {"current": [0.36, None, 0.36, 0.36]}, "null"),
        (lambda row: row | {"v_end": 3.5}, "v_end"),
        (lambda row: row | {"soh": float("nan")}, "soh"),
        (lambda row: row | {"cycle_index": 0}, "cycle_index"),
    ],
)
def test_validate_fragments_catches_malformed_rows(
    mutate: Callable[[dict[str, object]], dict[str, object]],
    message: str,
) -> None:
    """Каждый класс порчи фрагмента ловится с указанием строки."""
    frame = pl.DataFrame([mutate(_valid_fragment_row())], schema=dict(FRAGMENTS_SCHEMA))
    with pytest.raises(SchemaViolation, match=message):
        validate_fragments(frame, labeled=True, n_points=4)


def test_validate_fragments_catches_duplicate_id() -> None:
    """Дубли fragment_id — ошибка."""
    frame = pl.DataFrame(
        [_valid_fragment_row("same"), _valid_fragment_row("same")], schema=dict(FRAGMENTS_SCHEMA)
    )
    with pytest.raises(SchemaViolation, match="не уникален"):
        validate_fragments(frame, labeled=True, n_points=4)


# --------------------------------------------------------------------------
# CellBundle
# --------------------------------------------------------------------------


def test_bundle_rejects_foreign_cell_rows(synthetic_bundle: CellBundle) -> None:
    """Чужие cell_id внутри бандла — ошибка."""
    foreign = synthetic_bundle.timeseries.with_columns(pl.lit("oxford_Cell2").alias("cell_id"))
    broken = CellBundle(
        cells=synthetic_bundle.cells, cycles=synthetic_bundle.cycles, timeseries=foreign
    )
    with pytest.raises(SchemaViolation, match="чужие cell_id"):
        validate_bundle(broken)


def test_bundle_allows_timeseries_superset(synthetic_bundle: CellBundle) -> None:
    """§5.1: timeseries хранит все разобранные циклы, включая неразмеченные."""
    extra_cycle = synthetic_bundle.timeseries.filter(pl.col("cycle_index") == 6).with_columns(
        pl.lit(99, dtype=pl.Int32).alias("cycle_index")
    )
    bundle = CellBundle(
        cells=synthetic_bundle.cells,
        cycles=synthetic_bundle.cycles,
        timeseries=pl.concat([synthetic_bundle.timeseries, extra_cycle], how="vertical"),
    )
    warnings = validate_bundle(bundle)
    assert any("вне бенчмарка" in warning for warning in warnings)


def test_bundle_rejects_labelled_cycle_without_timeseries(synthetic_bundle: CellBundle) -> None:
    """Размеченный цикл без временного ряда — ошибка: фрагменты не построить."""
    trimmed = synthetic_bundle.timeseries.filter(pl.col("cycle_index") != 6)
    broken = CellBundle(
        cells=synthetic_bundle.cells, cycles=synthetic_bundle.cycles, timeseries=trimmed
    )
    with pytest.raises(SchemaViolation, match="отсутствуют в timeseries"):
        validate_bundle(broken)


def test_bundle_cell_requires_single_row(synthetic_bundle: CellBundle) -> None:
    """``CellBundle.cells`` — ровно одна строка."""
    doubled = pl.concat([synthetic_bundle.cells, synthetic_bundle.cells], how="vertical")
    broken = CellBundle(
        cells=doubled, cycles=synthetic_bundle.cycles, timeseries=synthetic_bundle.timeseries
    )
    with pytest.raises(SchemaViolation, match="одну строку"):
        _ = broken.cell
