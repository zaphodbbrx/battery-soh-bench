"""Унифицированная схема данных (ТЗ §4).

Модуль задаёт единственный контракт на форму данных, которой обмениваются
адаптеры (``bsb.ingest``), разметка (``bsb.labels``), генератор фрагментов
(``bsb.fragments``) и потребители (``bsb.dataset``, ``bsb.features``).

Два уровня parquet (``cells.parquet`` и ``cycles.parquet``) и партиционированный
по ``cell_id`` уровень ``timeseries/`` описаны словарями ``*_SCHEMA``.
Соглашения §4 (знак тока, ``charge_ah`` интегрированием, время от начала цикла,
пропуски как null) проверяются функциями ``validate_*``, а не остаются
договорённостью.

Значения controlled vocabulary заданы через ``polars.Enum``: движок отвергает
постороннее значение при сборке кадра, а не молча пропускает его в parquet.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Any, Final, Literal

import numpy as np
import polars as pl

__all__ = [
    "CELLS_SCHEMA",
    "CHEMISTRIES",
    "CHEMISTRY_DTYPE",
    "CYCLES_SCHEMA",
    "DATASETS",
    "DATASET_DTYPE",
    "FORM_FACTORS",
    "FORM_FACTOR_DTYPE",
    "FRAGMENTS_SCHEMA",
    "FRAGMENTS_UNLABELED_SCHEMA",
    "SOH_MAX",
    "SOH_MIN",
    "SOURCES",
    "SOURCE_DTYPE",
    "STEP_TYPES",
    "STEP_TYPE_DTYPE",
    "TIMESERIES_SCHEMA",
    "CellBundle",
    "Chemistry",
    "Dataset",
    "FormFactor",
    "SchemaViolation",
    "SohSource",
    "StepType",
    "check_structure",
    "conform",
    "empty_frame",
    "validate_bundle",
    "validate_cells",
    "validate_cycles",
    "validate_fragments",
    "validate_timeseries",
]

# --------------------------------------------------------------------------
# Controlled vocabularies (ТЗ §4.1-§4.4)
# --------------------------------------------------------------------------

#: Порядок категорий фиксирован: ``polars.Enum`` разных кадров склеивается
#: только при совпадении категорий, и это ровно то поведение, которое нужно.
DATASETS: Final[tuple[str, ...]] = ("calce", "kit", "oxford", "sandia")
CHEMISTRIES: Final[tuple[str, ...]] = ("LCO", "LFP", "NCA", "NMC", "NMC_SiO")
FORM_FACTORS: Final[tuple[str, ...]] = ("cylindrical_18650", "prismatic", "pouch")
SOURCES: Final[tuple[str, ...]] = ("measured", "interpolated")
STEP_TYPES: Final[tuple[str, ...]] = ("chg_cc", "chg_cv", "dch", "rest")

Dataset = Literal["calce", "kit", "oxford", "sandia"]
Chemistry = Literal["LCO", "LFP", "NCA", "NMC", "NMC_SiO"]
FormFactor = Literal["cylindrical_18650", "prismatic", "pouch"]
SohSource = Literal["measured", "interpolated"]
StepType = Literal["chg_cc", "chg_cv", "dch", "rest"]

DATASET_DTYPE: Final = pl.Enum(list(DATASETS))
CHEMISTRY_DTYPE: Final = pl.Enum(list(CHEMISTRIES))
FORM_FACTOR_DTYPE: Final = pl.Enum(list(FORM_FACTORS))
SOURCE_DTYPE: Final = pl.Enum(list(SOURCES))
STEP_TYPE_DTYPE: Final = pl.Enum(list(STEP_TYPES))

#: Границы целевой переменной (ТЗ §11 использует тот же диапазон для сабмита).
SOH_MIN: Final[float] = 0.0
SOH_MAX: Final[float] = 1.2

#: Санитарные границы физических величин. Это не параметры задачи, а защита от
#: протащенного мусора (нулей вместо пропусков, -1 как sentinel). Пороги
#: намеренно щедрые, чтобы не отвергать реальные данные.
PLAUSIBLE_BOUNDS: Final[Mapping[str, tuple[float, float]]] = {
    "voltage_v": (0.0, 10.0),
    "current_a": (-2000.0, 2000.0),
    "temperature_c": (-60.0, 200.0),
    "nominal_capacity_ah": (0.0, 1000.0),
    "bol_capacity_ah": (0.0, 1000.0),
    "capacity_ah": (0.0, 1000.0),
}

# --------------------------------------------------------------------------
# Схемы уровней (ТЗ §4.1-§4.3)
# --------------------------------------------------------------------------

#: ``cells.parquet`` — одна строка на ячейку.
CELLS_SCHEMA: Final[Mapping[str, pl.DataType]] = {
    "cell_id": pl.String,
    "dataset": DATASET_DTYPE,
    "chemistry": CHEMISTRY_DTYPE,
    "form_factor": FORM_FACTOR_DTYPE,
    "nominal_capacity_ah": pl.Float64,
    "bol_capacity_ah": pl.Float64,
    "temperature_setpoint_c": pl.Float64,
    "charge_protocol": pl.String,
    "batch_id": pl.String,
    "notes": pl.String,
}

#: ``cycles.parquet`` — одна строка на цикл.
CYCLES_SCHEMA: Final[Mapping[str, pl.DataType]] = {
    "cell_id": pl.String,
    "cycle_index": pl.Int32,
    "capacity_ah": pl.Float64,
    "is_reference": pl.Boolean,
    "soh": pl.Float64,
    "soh_source": SOURCE_DTYPE,
    "soh_gap_cycles": pl.Int32,
}

#: ``timeseries/`` — партиционирован по ``cell_id``.
TIMESERIES_SCHEMA: Final[Mapping[str, pl.DataType]] = {
    "cell_id": pl.String,
    "cycle_index": pl.Int32,
    "step_type": STEP_TYPE_DTYPE,
    "t_s": pl.Float64,
    "voltage_v": pl.Float64,
    "current_a": pl.Float64,
    "temperature_c": pl.Float64,
    "charge_ah": pl.Float64,
}

#: Внутренний уровень фрагментов: train/val, с метками (ТЗ §7).
FRAGMENTS_SCHEMA: Final[Mapping[str, pl.DataType]] = {
    "fragment_id": pl.String,
    "cell_id": pl.String,
    "cycle_index": pl.Int32,
    "v_start": pl.Float64,
    "v_end": pl.Float64,
    "q_norm": pl.List(pl.Float64),
    "current": pl.List(pl.Float64),
    "temperature": pl.List(pl.Float64),
    "soh": pl.Float64,
}

#: Тестовый файл участников: без ``cell_id``, ``cycle_index`` и ``soh`` (ТЗ §7).
FRAGMENTS_UNLABELED_SCHEMA: Final[Mapping[str, pl.DataType]] = {
    key: dtype
    for key, dtype in FRAGMENTS_SCHEMA.items()
    if key not in {"cell_id", "cycle_index", "soh"}
}


class SchemaViolation(ValueError):
    """Кадр не соответствует контракту §4. Сообщение указывает строки."""


# --------------------------------------------------------------------------
# CellBundle (ТЗ §6)
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CellBundle:
    """Три фрейма по схеме §4 для одной ячейки.

    ``cells`` содержит ровно одну строку. ``timeseries`` для KIT (ТЗ §6.3)
    наполняется адаптером потоково, но по-прежнему на одну ячейку.
    """

    cells: pl.DataFrame
    cycles: pl.DataFrame
    timeseries: pl.DataFrame

    @property
    def cell(self) -> dict[str, Any]:
        """Единственная строка ``cells`` как словарь."""
        if self.cells.height != 1:
            raise SchemaViolation(
                f"CellBundle.cells должен содержать ровно одну строку, получено {self.cells.height}"
            )
        return self.cells.row(0, named=True)

    @property
    def cell_id(self) -> str:
        """Идентификатор ячейки (``{dataset}_{original_id}``)."""
        return str(self.cell["cell_id"])


# --------------------------------------------------------------------------
# Структурные операции
# --------------------------------------------------------------------------


def empty_frame(schema: Mapping[str, pl.DataType]) -> pl.DataFrame:
    """Пустой кадр с точной схемой (порядок колонок сохраняется)."""
    return pl.DataFrame(schema=dict(schema))


def conform(
    df: pl.DataFrame, schema: Mapping[str, pl.DataType], *, name: str = "frame"
) -> pl.DataFrame:
    """Привести кадр к канонической схеме: отобрать, переставить, скастить.

    Адаптеры собирают кадры «как удобно» и вызывают ``conform`` на выходе,
    чтобы порядок колонок и типы не разъезжались между датасетами.

    Raises
    ------
    SchemaViolation
        Если колонки не хватает или приведение типа невозможно.
    """
    missing = [column for column in schema if column not in df.columns]
    if missing:
        raise SchemaViolation(f"{name}: отсутствуют колонки {missing}; есть {df.columns}")
    extra = [column for column in df.columns if column not in schema]
    if extra:
        raise SchemaViolation(f"{name}: лишние колонки {extra}")
    try:
        return df.select(
            [
                pl.col(column).cast(dtype, strict=True).alias(column)
                for column, dtype in schema.items()
            ]
        )
    except pl.exceptions.PolarsError as exc:  # pragma: no cover - сообщение важнее ветки
        raise SchemaViolation(f"{name}: не удалось привести типы к схеме §4: {exc}") from exc


def check_structure(
    df: pl.DataFrame,
    schema: Mapping[str, pl.DataType],
    *,
    name: str,
    check_order: bool = True,
) -> None:
    """Проверить наличие, порядок и типы колонок. Пустой кадр допустим."""
    problems: list[str] = []
    missing = [column for column in schema if column not in df.columns]
    extra = [column for column in df.columns if column not in schema]
    if missing:
        problems.append(f"отсутствуют колонки: {missing}")
    if extra:
        problems.append(f"лишние колонки: {extra}")
    for column, dtype in schema.items():
        if column not in df.columns:
            continue
        got = df.schema[column]
        if got != dtype:
            problems.append(f"колонка {column!r}: тип {got} вместо {dtype}")
    if check_order and not missing and not extra and list(df.columns) != list(schema):
        problems.append(f"порядок колонок {list(df.columns)} вместо {list(schema)}")
    if problems:
        raise SchemaViolation(f"{name}: нарушена схема §4: " + "; ".join(problems))


def _rows(df: pl.DataFrame, mask: pl.Series | pl.Expr, *, limit: int = 5) -> str:
    """Отпечаток первых нарушающих строк для сообщения об ошибке."""
    if not isinstance(mask, pl.Series):
        mask = df.select(mask.alias("_m"))["_m"]
    bad = df.filter(mask)
    head = bad.head(limit).to_dicts()
    suffix = "" if bad.height <= limit else f" … всего {bad.height}"
    return f"{head}{suffix}"


def _finite(df: pl.DataFrame, column: str) -> pl.Series:
    """Маска «значение не null и не NaN» для float-колонки."""
    return df[column].is_not_null() & df[column].is_finite()


def _out_of_bounds(df: pl.DataFrame, column: str) -> pl.Series:
    """Маска «значение заполнено и выходит за санитарные границы»."""
    low, high = PLAUSIBLE_BOUNDS[column]
    values = df[column]
    return values.is_not_null() & values.is_finite() & ((values < low) | (values > high))


def _duplicates(df: pl.DataFrame, keys: Sequence[str]) -> pl.Series:
    """Маска строк, дублирующихся по ``keys``."""
    return df.select(keys).is_duplicated()


def _list_missing_counts(df: pl.DataFrame, column: str) -> pl.Series:
    """Число null и NaN внутри list-колонки для каждой строки."""
    element = pl.element()
    return df.select(
        pl.col(column).list.eval(element.is_null() | element.is_nan()).list.sum().alias("n")
    )["n"]


# --------------------------------------------------------------------------
# Семантическая валидация (ТЗ §4.1)
# --------------------------------------------------------------------------


def validate_cells(df: pl.DataFrame, *, strict: bool = True) -> list[str]:
    """Проверить ``cells.parquet``. Возвращает список некритичных замечаний.

    Критично: схема, уникальность ``cell_id``, префикс ``{dataset}_``,
    положительные ёмкости, отсутствие null в обязательных полях.
    """
    check_structure(df, CELLS_SCHEMA, name="cells")
    if df.height and df["cell_id"].is_duplicated().any():
        raise SchemaViolation(
            f"cells: cell_id не уникален: {_rows(df, _duplicates(df, ['cell_id']))}"
        )
    null_id = df["cell_id"].is_null()
    if null_id.any():
        raise SchemaViolation(f"cells: пустой cell_id: {_rows(df, null_id)}")
    expected_prefix = pl.col("cell_id").str.split("_").list.first() == pl.col("dataset").cast(
        pl.String
    )
    if df.height and not bool(df.select(expected_prefix.all())[0, 0]):
        raise SchemaViolation(
            "cells: cell_id должен иметь вид {dataset}_{original_id}: "
            f"{_rows(df, ~expected_prefix)}"
        )
    for column in ("nominal_capacity_ah", "bol_capacity_ah"):
        bad = ~_finite(df, column) | (df[column] <= 0)
        if bad.any():
            raise SchemaViolation(f"cells: {column} должен быть > 0 и не null: {_rows(df, bad)}")
    for column in ("temperature_setpoint_c",):
        bad = df[column].is_null() | ~_finite(df, column)
        if bad.any():
            raise SchemaViolation(f"cells: {column} не должен быть null: {_rows(df, bad)}")
    for column in ("charge_protocol",):
        bad = df[column].is_null()
        if bad.any():
            raise SchemaViolation(f"cells: {column} не должен быть null: {_rows(df, bad)}")

    warnings: list[str] = []
    if strict:
        for column in ("nominal_capacity_ah", "bol_capacity_ah"):
            out = _out_of_bounds(df, column)
            if out.any():
                warnings.append(f"cells: {column} вне санитарных границ: {_rows(df, out, limit=3)}")
        ratio = df["bol_capacity_ah"] / df["nominal_capacity_ah"]
        odd = _finite(df, "bol_capacity_ah") & ((ratio > 1.5) | (ratio < 0.5))
        if odd.any():
            warnings.append(
                f"cells: bol_capacity_ah отличается от паспортной более чем в 1.5 раза: "
                f"{_rows(df, odd, limit=3)}"
            )
        bad_temp = (df["temperature_setpoint_c"] < -40) | (df["temperature_setpoint_c"] > 80)
        if bad_temp.any():
            warnings.append(
                f"cells: уставка температуры вне [-40, 80]: {_rows(df, bad_temp, limit=3)}"
            )
    return warnings


# --------------------------------------------------------------------------
# Семантическая валидация (ТЗ §4.2, §5)
# --------------------------------------------------------------------------


def validate_cycles(df: pl.DataFrame, *, strict: bool = True) -> list[str]:
    """Проверить ``cycles.parquet``, включая инварианты §5.

    Критично: схема, уникальность ``(cell_id, cycle_index)``,
    ``cycle_index >= 1``, ``soh`` в ``(0, 1.2]``, согласованность
    ``is_reference`` / ``soh_source`` / ``soh_gap_cycles``.
    """
    check_structure(df, CYCLES_SCHEMA, name="cycles")
    if df.height == 0:
        return []
    bad_id = df["cell_id"].is_null()
    if bad_id.any():
        raise SchemaViolation(f"cycles: пустой cell_id: {_rows(df, bad_id)}")
    dup = _duplicates(df, ["cell_id", "cycle_index"])
    if dup.any():
        raise SchemaViolation(f"cycles: дубли (cell_id, cycle_index): {_rows(df, dup)}")
    bad_index = df["cycle_index"] < 1
    if bad_index.any():
        raise SchemaViolation(f"cycles: cycle_index должен быть >= 1: {_rows(df, bad_index)}")
    bad_soh = ~_finite(df, "soh") | (df["soh"] <= SOH_MIN) | (df["soh"] > SOH_MAX)
    if bad_soh.any():
        raise SchemaViolation(
            f"cycles: soh должен лежать в ({SOH_MIN}, {SOH_MAX}]: {_rows(df, bad_soh)}"
        )
    bad_cap = df["capacity_ah"].is_not_null() & (
        ~_finite(df, "capacity_ah") | (df["capacity_ah"] <= 0)
    )
    if bad_cap.any():
        raise SchemaViolation(
            f"cycles: capacity_ah либо null, либо > 0 (нули вместо пропусков запрещены): "
            f"{_rows(df, bad_cap)}"
        )
    measured = pl.col("soh_source") == pl.lit("measured")
    interpolated = pl.col("soh_source") == pl.lit("interpolated")
    bad_measured = measured & (~pl.col("is_reference") | (pl.col("soh_gap_cycles") != 0))
    if bool(df.select(bad_measured.any())[0, 0]):
        raise SchemaViolation(
            "cycles: soh_source=measured требует is_reference=true и soh_gap_cycles=0: "
            f"{_rows(df, bad_measured)}"
        )
    bad_interp = interpolated & (pl.col("soh_gap_cycles") < 1)
    if bool(df.select(bad_interp.any())[0, 0]):
        raise SchemaViolation(
            f"cycles: soh_source=interpolated требует soh_gap_cycles >= 1: {_rows(df, bad_interp)}"
        )
    bad_gap = df["soh_gap_cycles"] < 0
    if bad_gap.any():
        raise SchemaViolation(f"cycles: soh_gap_cycles < 0: {_rows(df, bad_gap)}")
    bad_null_ref = df["is_reference"].is_null()
    if bad_null_ref.any():
        raise SchemaViolation(
            f"cycles: is_reference не должен быть null: {_rows(df, bad_null_ref)}"
        )

    warnings: list[str] = []
    if strict:
        over = df["soh"] > 1.0
        if over.any():
            warnings.append(
                f"cycles: soh > 1.0 (свежая ячейка выше паспорта) у {int(over.sum())} циклов: "
                f"{_rows(df, over, limit=3)}"
            )
    return warnings


# --------------------------------------------------------------------------
# Семантическая валидация (ТЗ §4.3)
# --------------------------------------------------------------------------


def validate_timeseries(
    df: pl.DataFrame,
    *,
    strict: bool = True,
    rest_current_tol_a: float = 0.05,
    current_sign_tol_a: float = 0.01,
    t_zero_tol_s: float = 1e-6,
    charge_first_tol_ah: float = 1e-6,
    integration_atol_ah: float = 2e-3,
    integration_rtol: float = 5e-3,
    require_nonempty: bool = True,
) -> list[str]:
    """Проверить ``timeseries/`` на соблюдение соглашений §4.3.

    Критично: схема, положительные напряжения, ``t_s`` от начала цикла и
    неубывание внутри цикла, ``charge_ah`` начинается с нуля и монотонна
    внутри шага, знак тока (заряд +, разряд -), rest около нуля.

    Возвращает замечания по точности интегрирования ``charge_ah``.
    """
    check_structure(df, TIMESERIES_SCHEMA, name="timeseries")
    if df.height == 0:
        if require_nonempty:
            raise SchemaViolation("timeseries: пустой кадр")
        return []
    bad_voltage = df["voltage_v"].is_null() | ~_finite(df, "voltage_v") | (df["voltage_v"] <= 0)
    if bad_voltage.any():
        raise SchemaViolation(
            f"timeseries: voltage_v должен быть > 0 и не null: {_rows(df, bad_voltage)}"
        )
    bad_current = df["current_a"].is_null() | ~_finite(df, "current_a")
    if bad_current.any():
        raise SchemaViolation(
            f"timeseries: current_a не должен быть null/NaN: {_rows(df, bad_current)}"
        )
    bad_t = df["t_s"].is_null() | ~_finite(df, "t_s") | (df["t_s"] < 0)
    if bad_t.any():
        raise SchemaViolation(f"timeseries: t_s должен быть >= 0: {_rows(df, bad_t)}")
    bad_charge = df["charge_ah"].is_null() | ~_finite(df, "charge_ah")
    if bad_charge.any():
        raise SchemaViolation(
            f"timeseries: charge_ah не должен быть null/NaN: {_rows(df, bad_charge)}"
        )

    warnings: list[str] = []
    for parts in df.partition_by(["cell_id", "cycle_index"], maintain_order=True):
        warnings.extend(
            _validate_cycle_steps(
                parts,
                strict=strict,
                rest_current_tol_a=rest_current_tol_a,
                current_sign_tol_a=current_sign_tol_a,
                t_zero_tol_s=t_zero_tol_s,
                charge_first_tol_ah=charge_first_tol_ah,
                integration_atol_ah=integration_atol_ah,
                integration_rtol=integration_rtol,
            )
        )
    return warnings


def _validate_cycle_steps(
    parts: pl.DataFrame,
    *,
    strict: bool,
    rest_current_tol_a: float,
    current_sign_tol_a: float,
    t_zero_tol_s: float,
    charge_first_tol_ah: float,
    integration_atol_ah: float,
    integration_rtol: float,
) -> list[str]:
    """Проверить один цикл: монотонность времени, шаги, знак тока, интеграл."""
    cell_id = str(parts["cell_id"][0])
    cycle_index = int(parts["cycle_index"][0])
    where = f"{cell_id} cycle {cycle_index}"
    t_s = parts["t_s"].to_numpy()
    current = parts["current_a"].to_numpy()
    charge = parts["charge_ah"].to_numpy()
    if (t_s[1:] < t_s[:-1]).any():
        idx = int((t_s[1:] < t_s[:-1]).argmax()) + 1
        raise SchemaViolation(
            f"timeseries: t_s убывает внутри цикла {where} в строке {idx} "
            f"({t_s[idx - 1]} -> {t_s[idx]})"
        )
    if t_s[0] > t_zero_tol_s:
        raise SchemaViolation(
            f"timeseries: t_s цикла {where} начинается с {t_s[0]}, а не с нуля (ТЗ §4.3)"
        )

    step_type = parts["step_type"].to_list()
    bounds = [0]
    for i in range(1, len(step_type)):
        if step_type[i] != step_type[i - 1]:
            bounds.append(i)
    bounds.append(len(step_type))

    warnings: list[str] = []
    for start, stop in pairwise(bounds):
        kind = step_type[start]
        segment_current = current[start:stop]
        segment_charge = charge[start:stop]
        segment_t = t_s[start:stop]
        if abs(segment_charge[0]) > charge_first_tol_ah:
            raise SchemaViolation(
                f"timeseries: charge_ah шага {kind} в {where} начинается с "
                f"{segment_charge[0]}, а не с нуля (ТЗ §4.3)"
            )
        finite = segment_current[~_isnan(segment_current)]
        worst = float(finite.max()) if finite.size else 0.0
        lowest = float(finite.min()) if finite.size else 0.0
        if kind in {"chg_cc", "chg_cv"} and lowest < -current_sign_tol_a:
            raise SchemaViolation(
                f"timeseries: шаг {kind} в {where} содержит ток {lowest:.4f} A <= 0 "
                "(ток заряда обязан быть положительным, ТЗ §4.3)"
            )
        if kind == "dch" and worst > current_sign_tol_a:
            raise SchemaViolation(
                f"timeseries: шаг dch в {where} содержит ток {worst:.4f} A >= 0 "
                "(ток разряда обязан быть отрицательным, ТЗ §4.3)"
            )
        if kind == "rest" and max(abs(worst), abs(lowest)) > rest_current_tol_a:
            raise SchemaViolation(
                f"timeseries: шаг rest в {where} содержит ток {max(abs(worst), abs(lowest)):.4f} A "
                f"(порог {rest_current_tol_a} A)"
            )
        deltas = segment_charge[1:] - segment_charge[:-1]
        if kind in {"chg_cc", "chg_cv"} and (deltas < -charge_first_tol_ah).any():
            raise SchemaViolation(f"timeseries: charge_ah убывает внутри шага {kind} в {where}")
        if kind == "dch" and (deltas > charge_first_tol_ah).any():
            raise SchemaViolation(f"timeseries: charge_ah растёт внутри шага dch в {where}")
        if strict and segment_t.size > 1:
            integrated = _cumulative_charge(segment_t, segment_current)
            scale = max(float(abs(segment_charge).max()), integration_atol_ah)
            deviation = float(abs(integrated - segment_charge).max())
            if deviation > integration_atol_ah + integration_rtol * scale:
                warnings.append(
                    f"timeseries: charge_ah шага {kind} в {where} расходится с интегралом тока "
                    f"на {deviation:.4f} A·ч (ТЗ §4.3 требует интегрирование)"
                )
    return warnings


def _isnan(values: Any) -> Any:
    """Маска NaN без зависимости от формы входа (numpy)."""
    return np.isnan(values)


def _cumulative_charge(t_s: Any, current: Any) -> Any:
    """Накопленный заряд трапециями от начала шага (в А·ч)."""
    deltas = np.diff(t_s) / 3600.0
    trapezoids = 0.5 * (current[1:] + current[:-1]) * deltas
    return np.concatenate([[0.0], np.cumsum(trapezoids)])


# --------------------------------------------------------------------------
# Семантическая валидация (ТЗ §7)
# --------------------------------------------------------------------------


def validate_fragments(
    df: pl.DataFrame,
    *,
    labeled: bool = True,
    n_points: int | None = None,
    strict: bool = True,
) -> list[str]:
    """Проверить ``fragments.parquet``.

    ``labeled=False`` — тестовый файл участников: без ``cell_id``,
    ``cycle_index`` и ``soh`` (ТЗ §7).
    """
    schema = FRAGMENTS_SCHEMA if labeled else FRAGMENTS_UNLABELED_SCHEMA
    name = "fragments" if labeled else "fragments (unlabeled)"
    check_structure(df, schema, name=name)
    if df.height == 0:
        return []
    dup = df["fragment_id"].is_duplicated()
    if dup.any():
        raise SchemaViolation(f"{name}: fragment_id не уникален: {_rows(df, dup)}")
    empty_id = df["fragment_id"].is_null() | (df["fragment_id"].str.len_chars() == 0)
    if empty_id.any():
        raise SchemaViolation(f"{name}: пустой fragment_id: {_rows(df, empty_id)}")
    for column in ("v_start", "v_end"):
        bad = ~_finite(df, column)
        if bad.any():
            raise SchemaViolation(f"{name}: {column} не должен быть null/NaN: {_rows(df, bad)}")
    bad_window = df["v_end"] <= df["v_start"]
    if bad_window.any():
        raise SchemaViolation(f"{name}: v_end должен быть > v_start: {_rows(df, bad_window)}")
    for column in ("q_norm", "current", "temperature"):
        nulls = _list_missing_counts(df, column)
        if (nulls > 0).any():
            bad = nulls > 0
            raise SchemaViolation(
                f"{name}: {column} содержит null/NaN (пропуски внутри фрагмента запрещены): "
                f"{_rows(df, bad)}"
            )
        lengths = df[column].list.len()
        if n_points is not None and (lengths != n_points).any():
            bad = lengths != n_points
            raise SchemaViolation(
                f"{name}: {column} должен иметь длину {n_points}, получено "
                f"{_rows(df.select(pl.col('fragment_id'), lengths.alias('len')), bad)}"
            )
    if labeled:
        bad_soh = ~_finite(df, "soh") | (df["soh"] <= SOH_MIN) | (df["soh"] > SOH_MAX)
        if bad_soh.any():
            raise SchemaViolation(
                f"{name}: soh должен лежать в ({SOH_MIN}, {SOH_MAX}]: {_rows(df, bad_soh)}"
            )
        bad_index = df["cycle_index"] < 1
        if bad_index.any():
            raise SchemaViolation(f"{name}: cycle_index должен быть >= 1: {_rows(df, bad_index)}")

    warnings: list[str] = []
    if strict:
        lengths = df["q_norm"].list.len().unique().to_list()
        if len(lengths) > 1:
            warnings.append(f"{name}: разная длина q_norm у фрагментов: {sorted(lengths)}")
    return warnings


# --------------------------------------------------------------------------
# CellBundle
# --------------------------------------------------------------------------


def validate_bundle(bundle: CellBundle, *, strict: bool = True) -> list[str]:
    """Проверить все три фрейма бандла и их взаимную согласованность.

    Дополнительно к §4 проверяется, что ``cycles`` и ``timeseries`` содержат
    ровно одну ячейку и что наборы ``cycle_index`` совпадают.
    """
    warnings = validate_cells(bundle.cells, strict=strict)
    if bundle.cells.height != 1:
        raise SchemaViolation(
            f"CellBundle.cells должен содержать одну строку, получено {bundle.cells.height}"
        )
    warnings += validate_cycles(bundle.cycles, strict=strict)
    warnings += validate_timeseries(bundle.timeseries, strict=strict)
    cell_id = bundle.cell_id
    foreign = set(bundle.cycles["cell_id"].to_list()) | set(bundle.timeseries["cell_id"].to_list())
    foreign -= {cell_id}
    if foreign:
        raise SchemaViolation(f"CellBundle {cell_id}: чужие cell_id в кадрах: {sorted(foreign)}")
    cycle_ids = set(bundle.cycles["cycle_index"].to_list())
    series_ids = set(bundle.timeseries["cycle_index"].to_list())
    # Каждый размеченный цикл обязан иметь временной ряд, иначе фрагменты не
    # построить. Обратное неверно: timeseries хранит все разобранные циклы, а
    # cycles — только те, что попали в бенчмарк (§5.1: рабочие циклы остаются
    # в хранилище, но метки не получают).
    if not cycle_ids <= series_ids:
        raise SchemaViolation(
            f"CellBundle {cell_id}: размеченные циклы отсутствуют в timeseries: "
            f"{sorted(cycle_ids - series_ids)[:10]}"
        )
    unlabelled = sorted(series_ids - cycle_ids)
    if unlabelled:
        warnings.append(
            f"CellBundle {cell_id}: в timeseries есть циклы, отсутствующие в cycles "
            f"(вне бенчмарка, ТЗ §5.1): {unlabelled[:5]}"
            + (f" (всего {len(unlabelled)})" if len(unlabelled) > 5 else "")
        )
    return warnings


def validate_all(
    frames: Iterable[tuple[str, pl.DataFrame]],
) -> dict[str, list[str]]:
    """Прогнать валидатор по именованным кадрам, вернуть замечания по имени."""
    validators = {
        "cells": validate_cells,
        "cycles": validate_cycles,
        "timeseries": validate_timeseries,
        "fragments": validate_fragments,
    }
    out: dict[str, list[str]] = {}
    for name, frame in frames:
        if name not in validators:
            raise SchemaViolation(f"неизвестный уровень данных: {name}")
        out[name] = validators[name](frame)
    return out
