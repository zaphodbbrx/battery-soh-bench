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

import os
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from itertools import pairwise
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
    "charge_throughput_by_cycle",
    "classify_step",
    "integrate_charge",
    "isolated_sign_flips",
    "plausible_mask",
    "segment_cycle",
    "split_into_steps",
    "suppress_native_stderr",
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


def charge_throughput_by_cycle(timeseries: pl.DataFrame) -> pl.DataFrame:
    """Перенесённый заряд каждого цикла, А·ч (сумма фаз заряда).

    ``charge_ah`` внутри фазы считается от её начала (§4.3), поэтому CC и CV
    надо складывать, а не брать общий максимум. Величина нужна правилу «полный
    цикл обязан перенести существенный заряд»: в CALCE и SNL встречаются циклы,
    которые доходят до границ по напряжению, но переносят почти ноль
    (ячейка уже была полной или запись оборвана), и их ёмкость занижена в разы.
    """
    return (
        timeseries.filter(pl.col("step_type").is_in(["chg_cc", "chg_cv"]))
        .group_by(["cycle_index", "step_type"])
        .agg(pl.col("charge_ah").max().alias("q_phase"))
        .group_by("cycle_index")
        .agg(pl.col("q_phase").sum().alias("q_charge"))
    )


@contextmanager
def suppress_native_stderr() -> Iterator[None]:
    """Заглушить stderr на время вызова нативной библиотеки.

    calamine (через fastexcel) пишет в stderr сообщения вида
    ``'chartsheet' not a valid worksheet`` для каждого файла, где есть лист
    диаграммы. В архиве Sandia таких файлов сотни, и шум перекрывает отчёт о
    разборе. Python-обёртка ``redirect_stderr`` на Rust не действует, поэтому
    перенаправляем файловый дескриптор.
    """
    saved = os.dup(2)
    try:
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            os.dup2(devnull.fileno(), 2)
            yield
    finally:
        os.dup2(saved, 2)
        os.close(saved)


def plausible_mask(
    t_s: Sequence[float],
    current_a: Sequence[float],
    voltage_v: Sequence[float],
    temperature_c: Sequence[float],
) -> np.ndarray:
    """Маска точек с физически осмысленными значениями (ТЗ §4.3).

    В сырых логах встречаются сенсорные сбои: в SNL это 5 точек из 11.6 млн с
    напряжением ``-0.005 В`` и ровно ``0 В``. §4.3 запрещает тащить суррогаты
    (нули и минус единицы вместо пропусков), а схема требует напряжение > 0,
    поэтому такие точки отбрасываются, а их число попадает в ``notes``.
    """
    from bsb.schema import PLAUSIBLE_BOUNDS

    time = np.asarray(t_s, dtype=float)
    current = np.asarray(current_a, dtype=float)
    voltage = np.asarray(voltage_v, dtype=float)
    temperature = np.asarray(temperature_c, dtype=float)
    low_v, high_v = PLAUSIBLE_BOUNDS["voltage_v"]
    low_i, high_i = PLAUSIBLE_BOUNDS["current_a"]
    low_t, high_t = PLAUSIBLE_BOUNDS["temperature_c"]
    finite = np.isfinite(time) & np.isfinite(current) & np.isfinite(voltage)
    # Температура в §4.3 nullable: у части ячеек SNL она не пишется вовсе
    # (весь цикл NaN). Пропуск должен становиться null, а не выбрасывать точки,
    # поэтому NaN здесь допустим, а отсекаются только выходы за границы.
    temperature_ok = ~np.isfinite(temperature) | ((temperature >= low_t) & (temperature <= high_t))
    return (
        finite
        & (time >= 0.0)
        & (voltage > low_v)
        & (voltage <= high_v)
        & (current >= low_i)
        & (current <= high_i)
        & temperature_ok
    )


def isolated_sign_flips(current_a: Sequence[float], *, tolerance: float) -> np.ndarray:
    """Маска одиночных отсчётов, знак которых противоречит обоим соседям.

    В SNL встречается точка ``-1.101 А`` посреди заряда на ``+1.499 А``: знак
    противоположен и предыдущему, и следующему отсчёту. Это сбой датчика, а не
    смена фазы: если такую точку оставить, она либо разрежет одну CC-фазу на
    две, либо даст ложное нарушение знака в §4.3.
    """
    current = np.asarray(current_a, dtype=float)
    if current.size < 3:
        return np.zeros(current.size, dtype=bool)
    sign = np.where(current > tolerance, 1, np.where(current < -tolerance, -1, 0))
    middle = sign[1:-1]
    flips = np.zeros(current.size, dtype=bool)
    flips[1:-1] = (
        (middle != 0) & (middle != sign[:-2]) & (middle != sign[2:]) & (sign[:-2] == sign[2:])
    )
    return flips


def split_into_steps(
    t_s: Sequence[float],
    current_a: Sequence[float],
    voltage_v: Sequence[float],
    temperature_c: Sequence[float],
    *,
    max_gap_s: float,
    min_points: int = 2,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Разбить запись на непрерывные шаги по разрывам времени.

    Разрыв больше ``max_gap_s`` означает перезапуск тестера или склейку двух
    записей (в Oxford это ``Cell1.cyc3200``). Неположительный шаг времени —
    тоже граница: в CALCE ``Test_Time(s)`` иногда сбрасывается посреди файла, и
    без деления такой сброс даёт отрицательный ``dt``, а с ним немонотонный
    ``charge_ah`` внутри шага.
    """
    time = np.asarray(t_s, dtype=float)
    current = np.asarray(current_a, dtype=float)
    voltage = np.asarray(voltage_v, dtype=float)
    temperature = np.asarray(temperature_c, dtype=float)
    if time.size < min_points:
        return []
    deltas = np.diff(time)
    breaks = np.flatnonzero((deltas > max_gap_s) | (deltas <= 0)) + 1
    bounds = [0, *breaks.tolist(), time.size]
    steps = []
    for start, stop in pairwise(bounds):
        if stop - start < min_points:
            continue
        steps.append(
            (
                time[start:stop] - time[start],
                current[start:stop],
                voltage[start:stop],
                temperature[start:stop],
            )
        )
    return steps


def segment_cycle(
    t_s: Sequence[float],
    current_a: Sequence[float],
    voltage_v: Sequence[float],
    temperature_c: Sequence[float],
    *,
    rest_current_tolerance_a: float,
    current_sign_tolerance_a: float,
    cv_voltage_tolerance_v: float,
    min_run: int = 3,
) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Разбить цикл на шаги по данным, когда исходник не даёт номеров шагов.

    Часть датасетов (BatteryLife/SNL) отдаёт внутрицикловые ряды без
    ``Step_Index``, поэтому границы шагов восстанавливаются по физике:

    1. точки делятся на режимы — отдых (``|i| <= rest_tol``), заряд, разряд;
    2. короткие пробеги (одиночные выбросы) приклеиваются к соседям;
    3. внутри заряда фаза CV отделяется по правилу §4.4: CV начинается там, где
       напряжение впервые поднялось в пределы ``cv_voltage_tolerance_v`` от
       максимума заряда, то есть дальше напряжение стоит, а ток падает.

    Возвращает список ``(step_type, t_s, current_a, voltage_v, temperature_c)``
    с временем от начала **шага**.
    """
    time = np.asarray(t_s, dtype=float)
    current = np.asarray(current_a, dtype=float)
    voltage = np.asarray(voltage_v, dtype=float)
    temperature = np.asarray(temperature_c, dtype=float)
    if time.size < 2:
        return []

    regime = np.where(
        current > current_sign_tolerance_a,
        1,
        np.where(current < -current_sign_tolerance_a, -1, 0),
    )
    bounds = [0, *(np.flatnonzero(np.diff(regime) != 0) + 1).tolist(), time.size]
    # Отдых — самостоятельный режим: если приклеивать нулевой ток к предыдущему,
    # то отдых после заряда «съедается» зарядом, и в chg_cc попадают точки с
    # нулевым током. Сливаются только короткие пробеги (сбои записи).
    runs: list[list[int]] = []
    for start, stop in pairwise(bounds):
        kind = int(regime[start])
        if stop - start >= min_run:
            runs.append([kind, start, stop])
            continue
        # Короткий пробег. Отдых безопасно приклеить к соседу: нулевой ток не
        # нарушает ни знак фазы, ни монотонность charge_ah. А вот короткий
        # всплеск противоположного знака приклеивать нельзя — в зарядную фазу
        # попадёт отрицательный ток (на реальных данных SNL так падало 15 ячеек).
        # Такой всплеск — сбой датчика, его точки выпадают из разметки.
        # Склейка допускается только при непрерывности (runs[-1][2] == start):
        # иначе диапазон фазы накроет уже выброшенный сбойный отсчёт, который
        # лежит между ними — так в chg_cv попадал ток -1.313 А.
        if kind == 0 and runs and runs[-1][2] == start:
            runs[-1][2] = stop
    while len(runs) > 1 and runs[0][2] - runs[0][1] < min_run:
        # Тот же принцип для ведущего короткого пробега: отдых можно отдать
        # следующей фазе, а всплеск знака — нет, иначе он попадёт в неё.
        if runs[0][0] == 0:
            runs[1][1] = runs[0][1]
        runs.pop(0)
    if not runs:
        return []

    pieces: list[tuple[str, int, int]] = []
    for kind_value, start, stop in runs:
        kind = kind_value
        if kind < 0:
            pieces.append(("dch", start, stop))
            continue
        if kind == 0:
            pieces.append(("rest", start, stop))
            continue
        # заряд: отделяем CV по напряжению (§4.4)
        segment_v = voltage[start:stop]
        plateau = float(np.max(segment_v)) - cv_voltage_tolerance_v
        plateau_hits = np.flatnonzero(segment_v >= plateau)
        if plateau_hits.size == 0 or plateau_hits[0] == 0:
            pieces.append(("chg_cv" if plateau_hits.size else "chg_cc", start, stop))
            continue
        split = start + int(plateau_hits[0])
        if split <= start or split >= stop:
            # Вырожденное деление: весь участок либо CC, либо CV. Пустой кусок
            # нельзя оставлять в списке — он разорвал бы склейку соседей.
            pieces.append(("chg_cv" if split <= start else "chg_cc", start, stop))
        else:
            pieces.append(("chg_cc", start, split))
            pieces.append(("chg_cv", split, stop))

    # Соседние однотипные фазы склеиваются: короткий провал записи режет одну
    # CC-фазу на две, и тогда charge_ah сбрасывался бы в её середине, а
    # валидатор §4.3 видел бы «заряд убывает». Слишком короткие куски при этом
    # надо отбросить ДО склейки, иначе они разделяют однотипных соседей.
    merged: list[tuple[str, int, int]] = []
    for kind, start, stop in pieces:
        if stop - start < min_run:
            continue
        if merged and merged[-1][0] == kind:
            merged[-1] = (kind, merged[-1][1], stop)
        else:
            merged.append((kind, start, stop))

    steps: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for kind, start, stop in merged:
        if stop - start < 2:
            continue
        t_local = time[start:stop] - time[start]
        steps.append(
            (kind, t_local, current[start:stop], voltage[start:stop], temperature[start:stop])
        )
    return steps


def classify_step(
    current_a: np.ndarray,
    voltage_v: np.ndarray,
    *,
    cc_current_tolerance_rel: float,
    cv_voltage_tolerance_v: float,
    rest_current_tolerance_a: float,
    current_sign_tolerance_a: float,
) -> str:
    """Определить ``step_type`` шага по §4.4 — единственное общее место.

    Правила ТЗ:

    * ``chg_cc`` — ток > 0 и стабилен в пределах ``cc_current_tolerance_rel``
      от медианы шага;
    * ``chg_cv`` — ток > 0, напряжение стабильно в пределах
      ``cv_voltage_tolerance_v``, ток падает;
    * ``dch`` — ток отрицательный;
    * ``rest`` — ток около нуля.

    Пороги приходят из ``step_labeling`` конфига, а не зашиты: на KIT с
    разрешением 2 с они почти наверняка потребуют правки (§4.4).
    """
    current = np.asarray(current_a, dtype=float)
    voltage = np.asarray(voltage_v, dtype=float)
    if current.size == 0:
        return "rest"
    median_current = float(np.median(current))
    if abs(median_current) <= rest_current_tolerance_a:
        return "rest"
    if median_current < -current_sign_tolerance_a:
        return "dch"
    if median_current <= current_sign_tolerance_a:
        return "rest"
    span_v = float(np.max(voltage) - np.min(voltage))
    stable_voltage = span_v <= cv_voltage_tolerance_v
    reference = abs(median_current) if median_current else 1.0
    within = np.abs(current - median_current) <= cc_current_tolerance_rel * reference
    stable_current = bool(within.mean() >= 0.5)
    current_falls = float(current[-1]) < float(current[0]) * (1.0 - cc_current_tolerance_rel)
    if stable_voltage and not stable_current:
        return "chg_cv"
    if current_falls and stable_voltage:
        return "chg_cv"
    return "chg_cc"


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
