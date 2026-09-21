"""Адаптер Sandia (SNL) — скрытый тест бенчмарка (ТЗ §6.4, §8, §17.4).

Источник
--------
Изначально §17.4 предполагал Battery Archive. Программного доступа там нет:
данные лежат в Postgres за Redash, анонимный API отвечает «Please login and try
again», массовая выгрузка — только по письму. Поэтому источник заменён на
открытый и воспроизводимый: подмножество SNL из **BatteryLife** (KDD 2025),
лицензия CC-BY-4.0, скачивается с Zenodo по прямой ссылке
(``scripts/download_sandia.sh``). Это тот же SNL: те же идентификаторы ячеек
(``SNL_18650_*``), тот же протокол (заряд 0.5C у всех ячеек), те же три химии.

Отличие от §17.4: 61 ячейка вместо 86. BatteryLife публикует ячейки, для
которых есть метки ресурса, поэтому группа 40-60 % SOC в подмножество не
попала. Для бенчмарка это скорее выигрыш: в §17.4 у части ячеек заряд шёл CC по
ёмкостным лимитам вместо CC-CV, то есть протокол заряда зависел от DOD и
смешивал перенос по химии с переносом по режиму (ровно то, от чего предостерегает
§5.1). Здесь 57 из 61 ячейки — окно 0-100 % SOC, и заряд единообразен.

Формат
------
Pickle на ячейку (``SNL_18650_NMC_25C_0-100_0.5-1C_a.pkl``): словарь с
метаданными и ``cycle_data`` — списком циклов, в каждом ряды ``current_in_A``,
``voltage_in_V``, ``charge_capacity_in_Ah``, ``discharge_capacity_in_Ah``,
``time_in_s``, ``temperature_in_C``. Нужна только стандартная библиотека,
pandas не требуется.

Что приходится восстанавливать самому:

* ``time_in_s`` кумулятивен по всему тесту, а §4.3 требует время от начала
  цикла — начало вычитается;
* номеров шагов в данных нет, поэтому ``step_type`` выделяется по физике
  (:func:`bsb.ingest.base.segment_cycle`), а CV отделяется от CC по §4.4;
* ``charge_ah`` считается интегрированием тока (§4.3), а не берётся из
  источника; исходный счётчик используется только для сверки.

Опорный цикл
------------
Правило владельца (``labels.reference_cycle_rules.sandia``): capacity check из
§17.4 — полный цикл 0-100 % SOC с разрядом на 0.5C. Флага проверки в данных нет,
но он выводится: рабочие циклы разряжаются на 1/2/3C, а проверки идут на 0.5C.
У восьми ячеек с рабочим режимом 0.5-0.5C проверкой становится любой полный
цикл, и это корректно: их рабочий режим и есть 0.5C.
"""

from __future__ import annotations

import pickle
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

import numpy as np
import polars as pl

from bsb.config import BenchmarkConfig
from bsb.ingest.base import (
    CellSource,
    RawBundle,
    build_raw_bundle,
    cycle_voltage_extremes,
    integrate_charge,
    isolated_sign_flips,
    plausible_mask,
    segment_cycle,
    validate_raw_bundle,
)
from bsb.labels import build_labels
from bsb.schema import CHEMISTRIES, CellBundle, SchemaViolation

__all__ = ["SandiaAdapter"]

#: Температура уставки в имени: ``..._NMC_25C_0-100_...``
TEMPERATURE_PATTERN: Final = re.compile(r"_(\d{2})C_")
#: Допуск на «дошёл до границы» по напряжению, В (границы берутся из метаданных).
DEFAULT_VOLTAGE_TOLERANCE_V: Final[float] = 0.05

#: Границы напряжения разряда 0-100 % SOC по химиям (ТЗ §17.4). В исходных
#: файлах они проставлены паспортно (``min_voltage_limit_in_V`` /
#: ``max_voltage_limit_in_V``), но в унифицированном уровне не хранятся, а
#: правило §5 должно переприменяться и к разобранным данным (``bsb labels``).
#: Совпадение с паспортными проверено на всех 57 ячейках SNL: разметка
#: опорных циклов совпадает цикл в цикл.
CHEMISTRY_VOLTAGE_LIMITS_V: Final[Mapping[str, tuple[float, float]]] = {
    "LFP": (2.0, 3.6),
    "NCA": (2.5, 4.2),
    "NMC": (2.0, 4.2),
}


class SandiaAdapter:
    """Адаптер подмножества SNL из BatteryLife."""

    name = "sandia"

    def __init__(self, config: BenchmarkConfig) -> None:
        """Сохранить конфиг (пороги §4.4 и правило опорных циклов)."""
        self.config = config

    # ------------------------------------------------------------------
    # discover
    # ------------------------------------------------------------------

    def discover(self, raw_dir: Path) -> list[CellSource]:
        """Найти ячейки: по одному pickle на ячейку, вложенность любая."""
        if not raw_dir.is_dir():
            raise FileNotFoundError(f"{raw_dir}: каталог сырых данных не найден")
        excluded = [
            str(window)
            for window in self.config.labels.rule(self.name).get("exclude_soc_windows", [])
        ]
        files = []
        for path in sorted(raw_dir.rglob("*.pkl")):
            if any(part.startswith("_") for part in path.parts):
                continue
            if any(f"_{window}_" in path.stem for window in excluded):
                continue
            files.append(path)
        if not files:
            raise FileNotFoundError(
                f"{raw_dir}: нет ни одного .pkl; сначала выполните scripts/download_sandia.sh"
            )
        return [CellSource(dataset=self.name, original_id=path.stem, path=path) for path in files]

    # ------------------------------------------------------------------
    # load_cell
    # ------------------------------------------------------------------

    def load_cell(self, src: CellSource) -> RawBundle:
        """Разобрать одну ячейку: циклы и внутрицикловые ряды по §4.3."""
        with src.path.open("rb") as handle:
            payload = pickle.load(handle)
        cycles_raw = payload.get("cycle_data") or []
        if not cycles_raw:
            raise SchemaViolation(f"{src.path.name}: нет cycle_data")

        limits = (
            float(payload.get("min_voltage_limit_in_V") or 0.0),
            float(payload.get("max_voltage_limit_in_V") or 0.0),
        )
        nominal = float(payload.get("nominal_capacity_in_Ah") or 0.0)
        labeling = self.config.step_labeling
        frames: list[pl.DataFrame] = []
        rows: list[dict[str, Any]] = []
        drifts: list[tuple[float, float]] = []
        notes: list[str] = []
        cycle_index = 0
        dropped_points = 0
        for cycle in cycles_raw:
            # Сенсорные сбои отбрасываем до сегментации, чтобы они не попали ни
            # в разметку шагов, ни в интегрирование заряда.
            current_raw = np.asarray(cycle["current_in_A"], dtype=float)
            mask = plausible_mask(
                cycle["time_in_s"],
                cycle["current_in_A"],
                cycle["voltage_in_V"],
                cycle["temperature_in_C"],
            )
            # Одиночные инвертированные отсчёты тока — сбой датчика, а не фаза.
            mask &= ~isolated_sign_flips(current_raw, tolerance=labeling.current_sign_tolerance_a)
            dropped_points += int((~mask).sum())  # нефизические значения
            if int(mask.sum()) < 2:
                notes.append(f"цикл {cycle.get('cycle_number')}: меньше двух пригодных точек")
                continue
            steps = segment_cycle(
                np.asarray(cycle["time_in_s"], dtype=float)[mask],
                np.asarray(cycle["current_in_A"], dtype=float)[mask],
                np.asarray(cycle["voltage_in_V"], dtype=float)[mask],
                np.asarray(cycle["temperature_in_C"], dtype=float)[mask],
                rest_current_tolerance_a=labeling.rest_current_tolerance_a,
                current_sign_tolerance_a=labeling.current_sign_tolerance_a,
                cv_voltage_tolerance_v=labeling.cv_voltage_tolerance_v,
            )
            if not steps:
                notes.append(f"цикл {cycle.get('cycle_number')}: шаги не выделены")
                continue
            cycle_index += 1
            offset = 0.0
            capacity: float | None = None
            parts: list[pl.DataFrame] = []
            kept_points = 0
            for kind, t_local, current, voltage, temperature in steps:
                charge = integrate_charge(t_local, current)
                parts.append(
                    pl.DataFrame(
                        {
                            "step_type": [kind] * t_local.size,
                            "t_s": offset + t_local,
                            "voltage_v": voltage,
                            "current_a": current,
                            "temperature_c": temperature,
                            "charge_ah": charge,
                        }
                    )
                )
                offset += float(t_local[-1]) + 1.0
                if kind == "dch":
                    capacity = abs(float(charge[-1]))
            kept_points = sum(len(part) for part in parts)
            dropped_points += int(mask.sum()) - kept_points
            frame = pl.concat(parts).with_columns(
                pl.lit(src.cell_id).alias("cell_id"),
                pl.lit(cycle_index, dtype=pl.Int32).alias("cycle_index"),
            )
            frames.append(frame)
            source_capacity = self._source_capacity(cycle)
            if capacity and source_capacity:
                drifts.append((capacity, source_capacity))
            rows.append(
                self._cycle_row(
                    src.cell_id,
                    cycle_index,
                    frame,
                    capacity=capacity,
                    limits=limits,
                    nominal=nominal,
                )
            )
        if not frames:
            raise SchemaViolation(f"{src.cell_id}: ни одного цикла с шагами не разобрано")
        # NaN в температуре — это пропуск, а не значение: §4.3 требует null.
        timeseries = pl.concat(frames).with_columns(
            pl.when(pl.col("temperature_c").is_finite())
            .then(pl.col("temperature_c"))
            .otherwise(None)
            .alias("temperature_c")
        )

        # Исходный счётчик ёмкости годится только для сверки: §4.3 требует
        # интегрировать ток самим. Заметное расхождение — повод для замечания.
        if dropped_points:
            notes.append(
                f"отброшено {dropped_points} точек: нефизические значения и одиночные "
                "всплески знака тока (сбои датчиков; ТЗ §4.3 запрещает суррогаты "
                "вместо пропусков)"
            )
        drift = self._capacity_drift(drifts)
        if drift is not None:
            notes.append(
                f"интеграл разрядной ёмкости расходится с исходным счётчиком "
                f"до {drift:.1%} (взято интегрирование, ТЗ §4.3)"
            )
        cell = {
            "cell_id": src.cell_id,
            "dataset": self.name,
            "chemistry": self._chemistry(payload, src),
            "form_factor": str(payload.get("form_factor") or "cylindrical_18650"),
            "nominal_capacity_ah": nominal,
            "temperature_setpoint_c": self._setpoint(src, timeseries),
            "charge_protocol": self._protocol(payload),
            "batch_id": None,
            "notes": self._notes(payload, notes),
        }
        return build_raw_bundle(cell, pl.DataFrame(rows), timeseries)

    @staticmethod
    def _source_capacity(cycle: dict[str, Any]) -> float | None:
        """Разрядная ёмкость цикла по исходному счётчику (для сверки)."""
        values = cycle.get("discharge_capacity_in_Ah") or []
        finite = [float(value) for value in values if value is not None]
        return max(finite) if finite else None

    @staticmethod
    def _capacity_drift(pairs: list[tuple[float, float]]) -> float | None:
        """Максимальное относительное расхождение интеграла и счётчика."""
        if not pairs:
            return None
        worst = max(abs(integrated - source) / source for integrated, source in pairs)
        return worst or None

    def _cycle_row(
        self,
        cell_id: str,
        cycle_index: int,
        frame: pl.DataFrame,
        *,
        capacity: float | None,
        limits: tuple[float, float],
        nominal: float,
    ) -> dict[str, Any]:
        """Строка ``cycles``: разрядная ёмкость и признак опорного цикла."""
        rule = self.config.labels.rule(self.name)
        tolerance = float(rule.get("voltage_tolerance_v", DEFAULT_VOLTAGE_TOLERANCE_V))
        rate_tolerance = float(rule.get("reference_rate_tolerance_rel", 0.02))
        charge = frame.filter(pl.col("step_type").is_in(["chg_cc", "chg_cv"]))
        discharge = frame.filter(pl.col("step_type") == pl.lit("dch"))
        is_reference = False
        if capacity and charge.height and discharge.height:
            low, high = limits
            reached_top = float(charge["voltage_v"].max()) >= high - tolerance
            reached_bottom = float(discharge["voltage_v"].min()) <= low + tolerance
            # Признак проверки — разряд на том же токе, что и заряд (0.5C/0.5C):
            # рабочие циклы разряжаются на 1/2/3C при том же заряде 0.5C.
            charge_current = abs(float(charge["current_a"].median()))
            discharge_current = abs(float(discharge["current_a"].median()))
            same_rate = (
                charge_current > 0
                and abs(discharge_current - charge_current) <= rate_tolerance * charge_current
            )
            is_reference = bool(reached_top and reached_bottom and same_rate)
        return {
            "cell_id": cell_id,
            "cycle_index": cycle_index,
            "capacity_ah": capacity,
            "is_reference": is_reference,
        }

    # ------------------------------------------------------------------
    # метаданные
    # ------------------------------------------------------------------

    @staticmethod
    def _chemistry(payload: dict[str, Any], src: CellSource) -> str:
        """Химия из метаданных; проверяется по справочнику §4.1."""
        value = str(payload.get("cathode_material") or "")
        if value in CHEMISTRIES:
            return value
        raise SchemaViolation(
            f"{src.cell_id}: химия {value!r} не входит в справочник {list(CHEMISTRIES)}"
        )

    @staticmethod
    def _setpoint(src: CellSource, timeseries: pl.DataFrame) -> float:
        """Уставка температуры: из имени ячейки, иначе медиана измеренной."""
        match = TEMPERATURE_PATTERN.search(src.original_id)
        if match:
            return float(match.group(1))
        median = timeseries["temperature_c"].median()
        if median is None:
            raise SchemaViolation(f"{src.cell_id}: температура не указана ни в имени, ни в данных")
        return float(median)

    @staticmethod
    def _protocol(payload: dict[str, Any]) -> str:
        """Собрать ``charge_protocol`` из метаданных (§4.1)."""

        def rates(key: str) -> str:
            entries = payload.get(key) or []
            values = [entry.get("rate_in_C") for entry in entries if isinstance(entry, dict)]
            described = [f"{value:g}C" for value in values if value]
            return "/".join(described) or "?"

        return (
            f"CC-CV заряд {rates('charge_protocol')} до "
            f"{payload.get('max_voltage_limit_in_V')} В, "
            f"разряд {rates('discharge_protocol')} до "
            f"{payload.get('min_voltage_limit_in_V')} В"
        )

    @staticmethod
    def _notes(payload: dict[str, Any], notes: list[str]) -> str:
        """Замечания разбора: источник, окно SOC, аномалии."""
        head = [
            "источник: BatteryLife (CC-BY-4.0), подмножество SNL",
            f"окно SOC {payload.get('SOC_interval')}, глубина разряда "
            f"{payload.get('depth_of_discharge')}",
        ]
        unique = sorted(set(notes))
        if unique:
            head.extend(unique[:5])
        return "; ".join(head)

    def _recompute_reference(self, bundle: RawBundle, rule: Mapping[str, Any]) -> pl.DataFrame:
        """Пересчитать ``is_reference``: полный размах и разряд на токе заряда.

        Признак capacity check в §17.4 — не флаг в данных, а совпадение: цикл
        прошёл 0-100 % SOC (до границ химии) и разряжался на том же токе, что и
        заряжался (0.5C). Рабочие циклы разряжаются на 1/2/3C при том же заряде
        0.5C, поэтому в проверки не попадают.
        """
        chemistry = str(bundle.cell["chemistry"])
        if chemistry not in CHEMISTRY_VOLTAGE_LIMITS_V:
            raise SchemaViolation(
                f"{bundle.cell_id}: для химии {chemistry!r} нет границ напряжения §17.4"
            )
        low, high = CHEMISTRY_VOLTAGE_LIMITS_V[chemistry]
        tolerance = float(rule.get("voltage_tolerance_v", DEFAULT_VOLTAGE_TOLERANCE_V))
        rate_tolerance = float(rule.get("reference_rate_tolerance_rel", 0.02))
        flags = cycle_voltage_extremes(bundle.timeseries).select(
            "cycle_index",
            (
                (pl.col("v_charge_max") >= high - tolerance)
                & (pl.col("v_discharge_min") <= low + tolerance)
                & (pl.col("i_charge") > 0)
                & (
                    (pl.col("i_discharge") - pl.col("i_charge")).abs()
                    <= rate_tolerance * pl.col("i_charge")
                )
            ).alias("_check"),
        )
        return (
            bundle.cycles.drop("is_reference")
            .join(flags, on="cycle_index", how="left")
            .with_columns(pl.col("_check").fill_null(False).alias("is_reference"))
            .drop("_check")
        )

    @staticmethod
    def _drop_shallow_checks(
        cycles: pl.DataFrame, bundle: RawBundle, *, rel: float
    ) -> tuple[pl.DataFrame, int]:
        """Снять признак опорного с проверок, перенёсших слишком мало заряда.

        Порог считается от ``bol`` — ёмкости первого capacity check: он снят на
        свежей ячейке, поэтому опорное значение берётся до фильтрации.
        """
        from bsb.ingest.base import charge_throughput_by_cycle

        candidates = cycles.filter(
            pl.col("is_reference") & pl.col("capacity_ah").is_not_null()
        ).sort("cycle_index")
        if candidates.height == 0:
            return cycles, 0
        bol = float(candidates["capacity_ah"][0])
        charge = charge_throughput_by_cycle(bundle.timeseries)
        joined = cycles.join(charge, on="cycle_index", how="left").with_columns(
            pl.col("q_charge").fill_null(0.0)
        )
        shallow = joined.filter(pl.col("is_reference") & (pl.col("q_charge") < rel * bol))[
            "cycle_index"
        ]
        excluded = sorted(int(value) for value in shallow.to_list())
        if not excluded:
            return cycles, 0
        return (
            cycles.with_columns(
                (pl.col("is_reference") & ~pl.col("cycle_index").is_in(excluded)).alias(
                    "is_reference"
                )
            ),
            len(excluded),
        )

    # ------------------------------------------------------------------
    # mark_reference_cycles
    # ------------------------------------------------------------------

    def mark_reference_cycles(self, bundle: RawBundle) -> CellBundle:
        """Применить правило capacity check (0-100 % на 0.5C) и посчитать §4.2.

        Признак проверки ёмкости пересчитывается из ``timeseries`` и химии
        ячейки, а не берётся из входного кадра: правило применяется и при
        разборе, и заново командой ``bsb labels``, а границы напряжения §17.4
        совпадают с паспортными, проставленными в исходных файлах (проверено
        на всех 57 ячейках SNL: разметка совпадает цикл в цикл).
        """
        warnings = validate_raw_bundle(bundle)
        rule = self.config.labels.rule(self.name)
        strategy = str(rule.get("strategy", "capacity_check"))
        if strategy != "capacity_check":
            raise NotImplementedError(
                f"sandia: стратегия {strategy!r} не реализована (ожидается capacity_check)"
            )
        cycles_frame = self._recompute_reference(bundle, rule)
        min_throughput_rel = float(rule.get("min_charge_throughput_rel", 0.0))
        if min_throughput_rel > 0:
            cycles_frame, shallow = self._drop_shallow_checks(
                cycles_frame, bundle, rel=min_throughput_rel
            )
            if shallow:
                warnings.append(
                    f"{bundle.cell_id}: {shallow} capacity check перенесли меньше "
                    f"{min_throughput_rel:.0%}·bol заряда (запись оборвана или ячейка "
                    "на пределе), в опорные не попали"
                )
        bundle = RawBundle(cells=bundle.cells, cycles=cycles_frame, timeseries=bundle.timeseries)
        reference = bundle.cycles.filter(
            pl.col("is_reference") & pl.col("capacity_ah").is_not_null()
        )
        if reference.height == 0:
            raise SchemaViolation(f"{bundle.cell_id}: ни одного capacity check — bol взять негде")
        skipped = bundle.cycles.height - reference.height
        if skipped:
            warnings.append(
                f"{bundle.cell_id}: {skipped} из {bundle.cycles.height} циклов не являются "
                "capacity check (полный 0-100 % при разряде 0.5C), в опорные не попали"
            )
        cycles = bundle.cycles
        if warnings:
            existing = bundle.cells["notes"][0]
            merged = "; ".join([part for part in (existing, *warnings) if part])
            bundle = RawBundle(
                cells=bundle.cells.with_columns(pl.lit(merged).alias("notes")),
                cycles=cycles,
                timeseries=bundle.timeseries,
            )
        else:
            bundle = RawBundle(cells=bundle.cells, cycles=cycles, timeseries=bundle.timeseries)
        return build_labels(bundle, self.config)
