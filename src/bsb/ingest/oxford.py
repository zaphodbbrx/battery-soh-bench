"""Адаптер Oxford Battery Degradation Dataset 1 (ТЗ §6.1, §17.3).

Датасет
-------
8 pouch-ячеек Kokam SLPB533459H4 (740 мА·ч), камера 40 °C, заряд CC-CV,
разряд по ездовому профилю Urban Artemis. Характеризация — каждые 100 рабочих
циклов. Источник: ORA, DOI 10.5287/bodleian:KO2kdmYGg, лицензия ODbL 1.0.

Что лежит в файле (сверено с ``Readme.txt`` и с самими данными)
--------------------------------------------------------------
``Oxford_Battery_Degradation_Dataset_1.mat`` — только характеризационные
измерения (рабочих циклов в наборе нет). Структура::

    Cell{1..8}.cyc{0000,0100,...,8200}.{C1ch,C1dc,OCVch,OCVdc}.{t,v,q,T}

* ``t`` — MATLAB datenum (дни), поэтому в §4.3 переводится в секунды от
  начала цикла;
* ``q`` — заряд в мА·ч со знаком (заряд положительный, разряд отрицательный);
* **поля тока в исходнике нет** — ``current_a`` восстанавливается
  дифференцированием ``q`` по ``t``; §4.3 запрещает брать ``charge_ah`` из
  счётчика исходника, поэтому здесь он считается интегрированием уже
  восстановленного тока;
* ``C1*`` — 1C (740 мА), ``OCV*`` — псевдо-OCV (40 мА = C/18.5). CV-фазы в
  характеризационных данных нет: обе записи заканчиваются на границе
  напряжения.

Опорный цикл
------------
В файле нет рабочих циклов, поэтому опорным является **каждый** цикл
характеризации (``strategy: characterization_cycle``). Дополнительно
проверяется, что разряд действительно медленный CC (порог
``discharge_current_tolerance_c``), — если это не так, ячейка получает
замечание в ``notes``.

Аномалия данных
---------------
В ``Cell1.cyc3200`` и ``Cell1.cyc4500`` шаги ``OCVch``/``OCVdc`` содержат два
склеенных сегмента записи, разделённых паузой в 6.3 суток (перезапуск
тестера). Шаг делится по разрыву времени, остаётся последний сегмент; факт
попадает в ``notes``.
"""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path
from typing import Any, Final

import numpy as np
import polars as pl
from scipy.io import loadmat

from bsb.config import BenchmarkConfig
from bsb.ingest.base import (
    CellSource,
    RawBundle,
    build_raw_bundle,
    integrate_charge,
    validate_raw_bundle,
)
from bsb.labels import build_labels
from bsb.schema import CellBundle, SchemaViolation

__all__ = ["OxfordAdapter"]

#: Паспортные данные ячейки (Readme.txt: 8 x Kokam SLPB533459H4, 740 мА·ч).
NOMINAL_CAPACITY_AH: Final[float] = 0.740
TEMPERATURE_SETPOINT_C: Final[float] = 40.0
#: Номинальные токи подшагов характеризации из Readme.txt, А.
NOMINAL_STEP_CURRENT_A: Final[dict[str, float]] = {
    "C1ch": 0.740,
    "C1dc": -0.740,
    "OCVch": 0.040,
    "OCVdc": -0.040,
}
#: Подшаги, которые вообще есть в характеризационном цикле.
CHARACTERISATION_STEPS: Final[tuple[str, ...]] = ("C1ch", "C1dc", "OCVch", "OCVdc")
#: Отклонение восстановленного тока от номинального, выше которого пишется замечание.
CURRENT_CROSSCHECK_REL: Final[float] = 0.20
#: Падение напряжения на стыке «конец заряда -> начало разряда», В. Больше
#: порога — предшествующий заряд неполный, и разрядная ёмкость занижена
#: (в наборе это 2 характеризации из 519: Cell2.cyc7400, Cell5.cyc5000).
JUNCTION_DROP_V: Final[float] = 0.05


class OxfordAdapter:
    """Адаптер Oxford Battery Degradation Dataset 1."""

    name = "oxford"

    def __init__(self, config: BenchmarkConfig) -> None:
        """Сохранить конфиг (выбор подшагов и пороги берутся только отсюда)."""
        self.config = config
        # Кэш на одну ячейку: файл 253 МБ, держать в памяти все восемь нельзя.
        self._cache_key: tuple[Path, str] | None = None
        self._cache_value: Any = None

    # ------------------------------------------------------------------
    # discover
    # ------------------------------------------------------------------

    def discover(self, raw_dir: Path) -> list[CellSource]:
        """Найти ячейки в каталоге сырых данных.

        Основной файл — самый большой ``.mat`` в каталоге: он содержит
        характеризацию всех ячеек. ``ExampleDC_C1.mat`` (первый цикл с
        Artemis-разрядом) намеренно игнорируется: в режиме «только опорные»
        (§5.1) рабочие циклы в бенчмарк не попадают.
        """
        if not raw_dir.is_dir():
            raise FileNotFoundError(f"{raw_dir}: каталог сырых данных не найден")
        candidates = sorted(
            raw_dir.glob("*.mat"), key=lambda path: path.stat().st_size, reverse=True
        )
        if not candidates:
            raise FileNotFoundError(
                f"{raw_dir}: нет ни одного .mat; сначала выполните scripts/download_oxford.sh"
            )
        main_file = candidates[0]
        cell_fields = self._cell_fields(main_file)
        if not cell_fields:
            raise SchemaViolation(f"{main_file}: не найдено ни одной ячейки Cell{{1..8}}")
        sources = [
            CellSource(dataset=self.name, original_id=name, path=main_file, extras={"field": name})
            for name in cell_fields
        ]
        return sources

    def _cell_fields(self, path: Path) -> list[str]:
        """Имена полей-ячеек в ``.mat`` (``Cell1``..``Cell8``)."""
        from scipy.io import whosmat

        names = [name for name, _shape, _kind in whosmat(path)]
        return sorted(name for name in names if name.startswith("Cell") and name[4:].isdigit())

    # ------------------------------------------------------------------
    # load_cell
    # ------------------------------------------------------------------

    def load_cell(self, src: CellSource) -> RawBundle:
        """Прочитать одну ячейку: ёмкости по циклам и timeseries по §4.3."""
        field = str(src.extras.get("field", src.original_id))
        cell_struct = self._load_cell_struct(src.path, field)
        cycle_fields = sorted(
            (
                name
                for name in cell_struct._fieldnames
                if name.startswith("cyc") and name[3:].isdigit()
            ),
            key=lambda name: int(name[3:]),
        )
        if not cycle_fields:
            raise SchemaViolation(f"{src.path}: у {field} нет ни одного цикла cycNNNN")

        oxford = self.config.ingest.oxford
        step_spec = (
            (oxford.charge_step, "chg_cc"),
            (oxford.discharge_step, "dch"),
        )
        frames: list[pl.DataFrame] = []
        rows: list[dict[str, Any]] = []
        notes: list[str] = []
        for cycle_field in cycle_fields:
            # cyc0000 — измерение до первого рабочего цикла; §4.2 требует
            # нумерацию с 1, поэтому индекс сдвинут на единицу.
            cycle_index = int(cycle_field[3:]) + 1
            cycle_struct = getattr(cell_struct, cycle_field)
            cycle_frames: list[pl.DataFrame] = []
            cursor = 0.0
            capacity_ah: float | None = None
            charge_final_voltage: float | None = None
            for step_name, step_type in step_spec:
                if step_name not in cycle_struct._fieldnames:
                    notes.append(f"{cycle_field}: нет подшага {step_name}")
                    continue
                step = getattr(cycle_struct, step_name)
                segments = self._split_segments(
                    np.asarray(step.t, dtype=float),
                    np.asarray(step.v, dtype=float),
                    np.asarray(step.q, dtype=float),
                    np.asarray(step.T, dtype=float),
                    max_gap_s=oxford.max_step_gap_s,
                )
                if not segments:
                    notes.append(f"{cycle_field}.{step_name}: пустая запись")
                    continue
                if len(segments) > 1:
                    notes.append(
                        f"{cycle_field}.{step_name}: {len(segments)} сегментов записи, "
                        "оставлен последний"
                    )
                t_s, voltage, current, temperature, charge = segments[-1]
                if step_type == "chg_cc":
                    charge_final_voltage = float(voltage[-1])
                elif charge_final_voltage is not None:
                    drop = charge_final_voltage - float(voltage[0])
                    if drop > JUNCTION_DROP_V:
                        notes.append(
                            f"{cycle_field}.{step_name}: разряд начат с {voltage[0]:.4f} В "
                            f"при конце заряда {charge_final_voltage:.4f} В "
                            f"(просадка {drop:.3f} В) — предшествующий заряд неполный, "
                            "разрядная ёмкость занижена"
                        )
                nominal = NOMINAL_STEP_CURRENT_A[step_name]
                median_current = float(np.median(current))
                if abs(median_current - nominal) > CURRENT_CROSSCHECK_REL * abs(nominal):
                    notes.append(
                        f"{cycle_field}.{step_name}: восстановленный ток "
                        f"{median_current * 1000:.1f} мА расходится с номиналом "
                        f"{nominal * 1000:.1f} мА более чем на "
                        f"{CURRENT_CROSSCHECK_REL:.0%}"
                    )
                cycle_frames.append(
                    pl.DataFrame(
                        {
                            "cell_id": [src.cell_id] * t_s.size,
                            "cycle_index": np.full(t_s.size, cycle_index, dtype=np.int32),
                            "step_type": [step_type] * t_s.size,
                            "t_s": cursor + t_s,
                            "voltage_v": voltage,
                            "current_a": current,
                            "temperature_c": temperature,
                            "charge_ah": charge,
                        }
                    )
                )
                cursor += float(t_s[-1]) + self._nominal_sample_interval(t_s)
                if step_type == "dch":
                    capacity_ah = abs(float(charge[-1]))
            if cycle_frames:
                frames.append(pl.concat(cycle_frames))
            rows.append(
                {
                    "cell_id": src.cell_id,
                    "cycle_index": cycle_index,
                    "capacity_ah": capacity_ah,
                    "is_reference": capacity_ah is not None,
                }
            )
        if not frames:
            raise SchemaViolation(f"{src.cell_id}: не удалось собрать ни одного шага timeseries")
        timeseries = pl.concat(frames)
        cycles = pl.DataFrame(rows)
        cell = {
            "cell_id": src.cell_id,
            "dataset": self.name,
            "chemistry": "LCO",
            "form_factor": "pouch",
            "nominal_capacity_ah": NOMINAL_CAPACITY_AH,
            "temperature_setpoint_c": TEMPERATURE_SETPOINT_C,
            "charge_protocol": (
                f"характеризация каждые 100 циклов: CC "
                f"{abs(NOMINAL_STEP_CURRENT_A[oxford.charge_step]) * 1000:.0f} мА "
                f"({abs(NOMINAL_STEP_CURRENT_A[oxford.charge_step]) / NOMINAL_CAPACITY_AH:.3f}C) "
                f"заряд до 4.2 В, CC разряд до 2.7 В"
            ),
            "batch_id": None,
            "notes": self._compose_notes(notes),
        }
        return build_raw_bundle(cell, cycles, timeseries)

    def _load_cell_struct(self, path: Path, field: str) -> Any:
        """Прочитать структуру одной ячейки из ``.mat``.

        В памяти держится не больше одной ячейки: ``variable_names`` ограничивает
        разбор, а кэш хранит только последний результат.
        """
        key = (path, field)
        if key != self._cache_key:
            data = loadmat(path, squeeze_me=True, struct_as_record=False, variable_names=[field])
            if field not in data:
                raise SchemaViolation(f"{path}: нет поля {field}")
            self._cache_key = key
            self._cache_value = data[field]
        return self._cache_value

    @staticmethod
    def _nominal_sample_interval(t_s: np.ndarray) -> float:
        """Типичный интервал дискретизации, с (для склейки шагов в цикл)."""
        if t_s.size < 2:
            return 0.0
        return float(np.median(np.diff(t_s)))

    @staticmethod
    def _split_segments(
        t_days: np.ndarray,
        voltage: np.ndarray,
        charge_mah: np.ndarray,
        temperature: np.ndarray,
        *,
        max_gap_s: float,
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """Разбить шаг на сегменты по разрывам времени и вернуть их по §4.3.

        Возвращает список кортежей ``(t_s, voltage_v, current_a, temperature_c,
        charge_ah)``: время от начала сегмента в секундах, ток в амперах
        (восстановлен дифференцированием заряда), заряд — интегрированием тока.
        """
        if t_days.size < 2:
            return []
        t_s_all = (t_days - t_days[0]) * 86400.0
        gaps = np.diff(t_s_all)
        breaks = np.flatnonzero(gaps > max_gap_s) + 1
        bounds = [0, *breaks.tolist(), t_days.size]
        segments = []
        for start, stop in pairwise(bounds):
            if stop - start < 2:
                continue
            t_s = t_s_all[start:stop] - t_s_all[start]
            v = voltage[start:stop]
            q_ah = charge_mah[start:stop] / 1000.0
            dt_s = np.diff(t_s)
            if np.any(dt_s <= 0):
                continue
            # Ток = производная заряда по времени (А·ч / ч).
            current = np.gradient(q_ah, t_s) * 3600.0
            charge_ah = integrate_charge(t_s, current)
            temperature_c = temperature[start:stop]
            segments.append((t_s, v, current, temperature_c, charge_ah))
        return segments

    @staticmethod
    def _compose_notes(notes: list[str]) -> str | None:
        """Собрать поле ``notes`` из замечаний разбора (ТЗ §4.1)."""
        if not notes:
            return None
        unique = sorted(set(notes))
        head = "; ".join(unique[:8])
        if len(unique) > 8:
            head += f"; … всего {len(unique)} замечаний"
        return head

    # ------------------------------------------------------------------
    # mark_reference_cycles
    # ------------------------------------------------------------------

    def mark_reference_cycles(self, bundle: RawBundle) -> CellBundle:
        """Разметить опорные циклы и посчитать §4.2 через :mod:`bsb.labels`.

        Правило ``characterization_cycle``: в основном файле набора рабочих
        циклов нет вовсе (§17.3), поэтому опорным является каждый цикл с
        измеренной разрядной ёмкостью. Порог
        ``labels.reference_cycle_rules.oxford.discharge_current_tolerance_c``
        используется как проверка, что разряд действительно медленный CC.
        """
        warnings = validate_raw_bundle(bundle)
        rule = self.config.labels.rule(self.name)
        strategy = str(rule.get("strategy", "characterization_cycle"))
        if strategy != "characterization_cycle":
            raise NotImplementedError(
                f"oxford: стратегия {strategy!r} не реализована "
                "(ожидается characterization_cycle, ТЗ §6.1)"
            )
        tolerance = float(rule.get("discharge_current_tolerance_c", 0.02))
        slow_discharge_a = abs(NOMINAL_STEP_CURRENT_A[self.config.ingest.oxford.discharge_step])
        discharge = bundle.timeseries.filter(pl.col("step_type") == pl.lit("dch"))
        if discharge.height:
            median_current = float(discharge["current_a"].median() or 0.0)
            if abs(abs(median_current) - slow_discharge_a) > tolerance * slow_discharge_a:
                warnings.append(
                    f"{bundle.cell_id}: средний ток разряда {median_current * 1000:.1f} мА "
                    f"отличается от C/18.5 ({slow_discharge_a * 1000:.1f} мА) больше чем на "
                    f"{tolerance:.0%}"
                )
        reference = bundle.cycles.filter(
            pl.col("is_reference") & pl.col("capacity_ah").is_not_null()
        )
        if reference.height == 0:
            raise SchemaViolation(f"{bundle.cell_id}: нет опорных циклов с измеренной ёмкостью")
        if warnings:
            existing = bundle.cells["notes"][0]
            merged = "; ".join([part for part in (existing, *warnings) if part])
            bundle = RawBundle(
                cells=bundle.cells.with_columns(pl.lit(merged).alias("notes")),
                cycles=bundle.cycles,
                timeseries=bundle.timeseries,
            )
        return build_labels(bundle, self.config)
