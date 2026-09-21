"""Адаптер CALCE (ТЗ §6.2, §17.1) — фаза 4 плана §15.

Данные
------
Серия CS2 — призматические LCO 1.1 А·ч; температура комнатная (в логах не
пишется). Заряд у **всех** ячеек одинаковый: CC 0.5C до 4.2 В, затем CV до тока
0.05 А. Именно постоянство заряда делает серию пригодной для бенчмарка (§6.2),
потому что фрагмент строится из зарядной кривой.

Различается только разряд, по типам:

======  =====================================================
тип 1   CC 0.5C (CS2_8, CS2_21, CS2_33, CS2_34)
тип 2   CC 1C (CS2_35..CS2_38)
тип 3   чередование токов разряда 0.11-2.2 А (CS2_3, CS2_9)
тип 4   случайная отсечка по напряжению (CS2_7)
тип 5   частичные циклы 3.77-2.7 В (CS2_5, CS2_6)
тип 6   частичные циклы 4.2-3.77 В (CS2_24, CS2_25)
======  =====================================================

Два формата
-----------
* **Arbin** (13 ячеек) — Excel, один файл на дату тестирования, лист данных с
  колонками ``Test_Time(s)``, ``Step_Index``, ``Current(A)``, ``Voltage(V)``,
  ``Charge_Capacity(Ah)``, ``Discharge_Capacity(Ah)``; дискретизация 30 с.
* **CADEX** (CS2_8, CS2_21) — ``.txt`` с табуляцией; ``Time`` в **минутах**
  (``Duration`` — секунды с начала программы), ``mV`` и ``mA``; дискретизация
  60 с. Заряд и разряд идут вперемешку короткими блоками, поэтому цикл
  собирается как пара «блок заряда + следующий за ним блок разряда».

Шаги и циклы определяются по данным, а не по номерам шагов программы:
``step_type`` размечает общий :func:`bsb.ingest.base.classify_step` по §4.4, а
цикл — это последовательность «заряд(ы) + разряд», закрывающаяся разрядом. Так
один и тот же код обслуживает Arbin и CADEX и не зависит от нумерации
программы тестера.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from itertools import pairwise
from pathlib import Path
from typing import Any, Final

import fastexcel
import numpy as np
import polars as pl

from bsb.config import BenchmarkConfig
from bsb.ingest.base import (
    CellSource,
    RawBundle,
    build_raw_bundle,
    charge_throughput_by_cycle,
    classify_step,
    cycle_voltage_extremes,
    integrate_charge,
    isolated_sign_flips,
    plausible_mask,
    split_into_steps,
    suppress_native_stderr,
    validate_raw_bundle,
)
from bsb.labels import build_labels
from bsb.schema import CellBundle, SchemaViolation

__all__ = ["CalceAdapter"]

#: Паспортные данные CS2 (ТЗ §17.1): призматические LCO 1.1 А·ч.
NOMINAL_CAPACITY_AH: Final[float] = 1.1
#: Температура: в логах не пишется, по документации испытания комнатные.
TEMPERATURE_SETPOINT_C: Final[float] = 25.0
#: Имя файла: ``CS2_33_10_05_10.xlsx`` -> серия CS2, ячейка 33.
CELL_PATTERN: Final = re.compile(r"^(?P<series>CS2|CX2)_(?P<number>\d+)")
#: Разрыв внутри файла (перезапуск тестера); между файлами паузы заведомо больше.
DEFAULT_MAX_GAP_S: Final[float] = 600.0


class CalceAdapter:
    """Адаптер CALCE CS2 (Arbin Excel + CADEX txt)."""

    name = "calce"

    def __init__(self, config: BenchmarkConfig) -> None:
        """Сохранить конфиг (серии, пороги §4.4, правило опорных циклов)."""
        self.config = config

    # ------------------------------------------------------------------
    # discover
    # ------------------------------------------------------------------

    def discover(self, raw_dir: Path) -> list[CellSource]:
        """Найти ячейки выбранных серий в каталоге сырых данных.

        Каталог может быть любой вложенности: распакованные архивы CALCE кладут
        файлы в ``<cell>/<cell>/``. Идентификатор ячейки берётся из имени файла.
        """
        if not raw_dir.is_dir():
            raise FileNotFoundError(f"{raw_dir}: каталог сырых данных не найден")
        series = set(self.config.ingest.calce.series)
        grouped: dict[str, list[Path]] = {}
        for path in sorted(raw_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in {".xlsx", ".xls", ".txt"}:
                continue
            match = CELL_PATTERN.match(path.name)
            if match is None or match.group("series") not in series:
                continue
            cell_id = f"{match.group('series')}_{match.group('number')}"
            grouped.setdefault(cell_id, []).append(path)
        if not grouped:
            raise FileNotFoundError(
                f"{raw_dir}: не найдено ячеек серий {sorted(series)}; "
                "сначала выполните scripts/download_calce.sh"
            )
        # Решение владельца (§18 п.5): ячейки с меняющимся током разряда в
        # бенчмарк не входят — их таргет зависит от режима, а не от старения.
        excluded = {
            str(item) for item in self.config.labels.rule(self.name).get("exclude_cells", [])
        }
        grouped = {key: value for key, value in grouped.items() if key not in excluded}
        if not grouped:
            raise FileNotFoundError(
                f"{raw_dir}: все ячейки серий {sorted(series)} исключены правилом exclude_cells"
            )
        return [
            CellSource(
                dataset=self.name,
                original_id=cell_id,
                path=Path(min(files)),
                extras={"files": sorted(files)},
            )
            for cell_id, files in sorted(grouped.items())
        ]

    # ------------------------------------------------------------------
    # load_cell
    # ------------------------------------------------------------------

    def load_cell(self, src: CellSource) -> RawBundle:
        """Прочитать одну ячейку: склеить все файлы в один временной ряд.

        Файлы обрабатываются по возрастанию даты первой записи, ``cycle_index``
        — сплошной счётчик по ячейке (§4.2).
        """
        files = [Path(item) for item in src.extras.get("files", [src.path])]
        recordings: list[tuple[float, Path, pl.DataFrame]] = []
        notes: list[str] = []
        for path in files:
            frame = self._read_file(path)
            if frame.height == 0:
                notes.append(f"{path.name}: нет данных")
                continue
            recordings.append((float(frame["date"].max()), path, frame))
        if not recordings:
            raise SchemaViolation(f"{src.cell_id}: ни один файл не дал данных")
        recordings.sort(key=lambda item: item[0])

        step_frames: list[pl.DataFrame] = []
        cycle_rows: list[dict[str, Any]] = []
        dropped_points = 0
        cycle_index = 0
        for _date, path, frame in recordings:
            steps, dropped = self._steps_of(frame, max_gap_s=DEFAULT_MAX_GAP_S)
            dropped_points += dropped
            if not steps:
                notes.append(f"{path.name}: шаги не выделены")
                continue
            open_steps: list[pl.DataFrame] = []
            for step in steps:
                kind = self._classify(step)
                step = step.with_columns(pl.lit(kind).alias("step_type"))
                if not open_steps and kind == "rest":
                    continue
                open_steps.append(step)
                if kind == "dch":
                    cycle_index += 1
                    cycle = self._merge_cycle(
                        open_steps, cell_id=src.cell_id, cycle_index=cycle_index
                    )
                    step_frames.append(cycle)
                    cycle_rows.append(self._cycle_row(src.cell_id, cycle_index, cycle))
                    open_steps = []
            if open_steps:
                notes.append(f"{path.name}: хвост без разряда, пропущен")
        if dropped_points:
            notes.append(
                f"отброшено {dropped_points} точек: пустые строки и одиночные всплески "
                "знака тока (ТЗ §4.3 запрещает пропуски и суррогаты)"
            )
        if not step_frames:
            raise SchemaViolation(f"{src.cell_id}: не удалось собрать ни одного полного цикла")
        timeseries = pl.concat(step_frames)
        cycles = pl.DataFrame(cycle_rows)
        cell = self._cell_dict(src, timeseries, files=files, notes=notes)
        return build_raw_bundle(cell, cycles, timeseries)

    def _cell_dict(
        self,
        src: CellSource,
        timeseries: pl.DataFrame,
        *,
        files: list[Path],
        notes: list[str],
    ) -> dict[str, Any]:
        """Метаданные ячейки по §4.1. Отдельный метод — для переиспользования."""
        return {
            "cell_id": src.cell_id,
            "dataset": self.name,
            "chemistry": "LCO",
            "form_factor": "prismatic",
            "nominal_capacity_ah": NOMINAL_CAPACITY_AH,
            "temperature_setpoint_c": TEMPERATURE_SETPOINT_C,
            "charge_protocol": self._protocol(timeseries),
            "batch_id": None,
            "notes": self._notes(files, notes),
        }

    # ------------------------------------------------------------------
    # чтение форматов
    # ------------------------------------------------------------------

    #: Сигнатуры форматов по содержимому: расширениям в CALCE верить нельзя
    #: (``CS2_5_5_15_12.xls`` — это на самом деле xlsx).
    ZIP_SIGNATURE: Final[bytes] = b"PK\x03\x04"
    OLE_SIGNATURE: Final[bytes] = b"\xd0\xcf\x11\xe0"

    def _read_file(self, path: Path) -> pl.DataFrame:
        """Прочитать файл Arbin (Excel) или CADEX (txt) в общий вид.

        Формат определяется по сигнатуре, а не по расширению: в серии CS2
        встречаются xlsx-файлы с расширением ``.xls``.

        Общий вид: ``date`` (порядок записи), ``t_s``, ``current_a``,
        ``voltage_v``, ``temperature_c``, ``step_key`` — ключ непрерывного шага.
        """
        with path.open("rb") as handle:
            signature = handle.read(4)
        if signature in {self.ZIP_SIGNATURE, self.OLE_SIGNATURE}:
            return self._read_arbin(path)
        return self._read_cadex(path)

    #: Колонки, по которым лист опознаётся как лог данных, а не сводка.
    ARBIN_REQUIRED_COLUMNS: Final[frozenset[str]] = frozenset(
        {"Test_Time(s)", "Step_Index", "Current(A)", "Voltage(V)"}
    )

    def _read_arbin(self, path: Path) -> pl.DataFrame:
        """Листы-логи данных Arbin.

        Кроме листа ``Info`` с метаданными в файлах встречается лист
        ``Statistics_*`` — построчная сводка по циклам. Она полезна для сверки,
        но это не лог, поэтому листы отбираются по колонкам, а не по имени:
        имена листов у CALCE непостоянны.
        """
        # Читаем байты, а не путь: calamine иначе доверится расширению и
        # откажется открывать xlsx, названный .xls.
        with suppress_native_stderr():
            reader = fastexcel.read_excel(path.read_bytes())
        frames: list[pl.DataFrame] = []
        for name in reader.sheet_names:
            if name == "Info":
                continue
            with suppress_native_stderr():
                table = reader.load_sheet(name).to_polars()
            if set(table.columns) >= self.ARBIN_REQUIRED_COLUMNS:
                frames.append(self._normalize_arbin(table, path))
        if not frames:
            raise SchemaViolation(
                f"{path}: нет листа с колонками {sorted(self.ARBIN_REQUIRED_COLUMNS)} "
                f"(листы: {reader.sheet_names})"
            )
        return pl.concat(frames, how="vertical")

    def _normalize_arbin(self, frame: pl.DataFrame, path: Path) -> pl.DataFrame:
        """Привести колонки Arbin к общему виду."""
        missing = self.ARBIN_REQUIRED_COLUMNS - set(frame.columns)
        if missing:
            raise SchemaViolation(f"{path}: в листе нет колонок {sorted(missing)}")
        date = (
            frame["Date_Time"].cast(pl.Int64).cast(pl.Float64)
            if "Date_Time" in frame.columns
            else pl.Series(np.arange(frame.height), dtype=pl.Float64)
        )
        return pl.DataFrame(
            {
                "date": date,
                "t_s": frame["Test_Time(s)"].cast(pl.Float64),
                "current_a": frame["Current(A)"].cast(pl.Float64),
                "voltage_v": frame["Voltage(V)"].cast(pl.Float64),
                "step_key": frame["Step_Index"].cast(pl.Int64),
            }
        )

    def _read_cadex(self, path: Path) -> pl.DataFrame:
        """CADEX ``.txt``: табуляция, ``Time`` в минутах, ``mV`` и ``mA``.

        Единицы проверены по данным: ``Time`` × 60 совпадает с ``Duration``
        (секунды с начала программы), то есть дискретизация здесь 60 с.
        """
        frame = pl.read_csv(path, separator="\t", infer_schema_length=5000)
        frame = frame.select([name for name in frame.columns if name])
        required = {"Time", "mV", "mA", "Pgm step"}
        missing = required - set(frame.columns)
        if missing:
            raise SchemaViolation(f"{path}: в CADEX-файле нет колонок {sorted(missing)}")
        return pl.DataFrame(
            {
                "date": pl.Series(np.arange(frame.height), dtype=pl.Float64),
                "t_s": frame["Time"].cast(pl.Float64) * 60.0,
                "current_a": frame["mA"].cast(pl.Float64) / 1000.0,
                "voltage_v": frame["mV"].cast(pl.Float64) / 1000.0,
                "step_key": frame["Pgm step"].cast(pl.Int64),
            }
        )

    # ------------------------------------------------------------------
    # шаги и циклы
    # ------------------------------------------------------------------

    def _steps_of(self, frame: pl.DataFrame, *, max_gap_s: float) -> tuple[list[pl.DataFrame], int]:
        """Разбить запись файла на непрерывные шаги.

        Границы: смена ``step_key`` программы и разрывы времени (перезапуск
        тестера оставляет в файле склеенные куски — как в Oxford).

        Возвращает шаги и число отброшенных точек: в архиве Sandia в листах
        встречаются полностью пустые строки (109 из 11563 в ``LCO_1_25C_Reg``),
        а §4.3 запрещает протаскивать пропуски и суррогаты дальше.
        """
        if "temperature_c" not in frame.columns:
            frame = frame.with_columns(pl.lit(None, dtype=pl.Float64).alias("temperature_c"))
        blocks = frame.with_columns(
            (
                (pl.col("step_key") != pl.col("step_key").shift())
                | (pl.col("t_s").diff() > max_gap_s)
            )
            .cum_sum()
            .alias("_block")
        ).partition_by("_block", maintain_order=True)
        steps: list[pl.DataFrame] = []
        dropped = 0
        for block in blocks:
            # Внутри шага программы тестер иногда успевает зарядить и разрядить
            # (в CS2_36 в одном Step_Index 7% точек идут с обратным знаком).
            # Такие участки делятся по устойчивой смене знака тока, иначе
            # step_type перестаёт быть свойством шага.
            bounds = [
                0,
                *self._sign_boundaries(
                    block["current_a"].to_numpy(),
                    tolerance=self.config.step_labeling.current_sign_tolerance_a,
                ),
                block.height,
            ]
            for start, stop in pairwise(bounds):
                piece = block[start:stop]
                if piece.height < 2:
                    continue
                # Пустые строки листа и одиночные всплески знака тока — сбои
                # записи; §4.3 требует не тащить их дальше.
                mask = plausible_mask(
                    piece["t_s"].to_numpy(),
                    piece["current_a"].to_numpy(),
                    piece["voltage_v"].to_numpy(),
                    piece["temperature_c"].to_numpy(),
                ) & ~isolated_sign_flips(
                    piece["current_a"].to_numpy(),
                    tolerance=self.config.step_labeling.current_sign_tolerance_a,
                )
                dropped += int((~mask).sum())
                if int(mask.sum()) < 2:
                    continue
                piece = piece.filter(pl.Series(mask))
                pieces = split_into_steps(
                    piece["t_s"].to_numpy(),
                    piece["current_a"].to_numpy(),
                    piece["voltage_v"].to_numpy(),
                    piece["temperature_c"].to_numpy(),
                    max_gap_s=max_gap_s,
                )
                for t_s, current, voltage, temperature in pieces:
                    steps.append(
                        pl.DataFrame(
                            {
                                "t_s": t_s,
                                "current_a": current,
                                "voltage_v": voltage,
                                "temperature_c": temperature,
                            }
                        )
                    )
        return steps, dropped

    @staticmethod
    def _sign_boundaries(current: np.ndarray, *, tolerance: float, min_run: int = 3) -> list[int]:
        """Индексы устойчивой смены знака тока внутри шага программы.

        Короткий выброс знака не считается границей: новый знак должен
        продержаться хотя бы ``min_run`` точек.
        """
        boundaries: list[int] = []
        sign = np.where(current > tolerance, 1, np.where(current < -tolerance, -1, 0))
        for index in range(1, sign.size):
            previous = sign[index - 1]
            if previous == 0 or sign[index] == 0 or sign[index] == previous:
                continue
            run = sign[index : index + min_run]
            if run.size and float((run == sign[index]).mean()) > 0.5:
                boundaries.append(index)
        return boundaries

    def _classify(self, step: pl.DataFrame) -> str:
        """Разметить шаг общим классификатором §4.4."""
        labeling = self.config.step_labeling
        return classify_step(
            step["current_a"].to_numpy(),
            step["voltage_v"].to_numpy(),
            cc_current_tolerance_rel=labeling.cc_current_tolerance_rel,
            cv_voltage_tolerance_v=labeling.cv_voltage_tolerance_v,
            rest_current_tolerance_a=labeling.rest_current_tolerance_a,
            current_sign_tolerance_a=labeling.current_sign_tolerance_a,
        )

    def _merge_cycle(
        self,
        steps: list[pl.DataFrame],
        *,
        cell_id: str,
        cycle_index: int,
    ) -> pl.DataFrame:
        """Собрать цикл из шагов, склеивая соседние однотипные в одну фазу.

        §4.3 определяет шаг как участок с одним ``step_type``, а тестер может
        выдать подряд два шага программы одного типа (в CS2 так бывает у CV).
        Тогда это одна фаза: время и ``charge_ah`` в ней непрерывны, а не
        сбрасываются в середине.

        ``t_s`` отсчитывается от начала **цикла** (§4.3), поэтому у каждого
        цикла он начинается с нуля.
        """
        groups: list[list[pl.DataFrame]] = []
        for step in steps:
            kind = str(step["step_type"][0])
            if groups and str(groups[-1][0]["step_type"][0]) == kind:
                groups[-1].append(step)
            else:
                groups.append([step])

        frames: list[pl.DataFrame] = []
        offset = 0.0
        for group in groups:
            kind = str(group[0]["step_type"][0])
            t_parts: list[np.ndarray] = []
            current_parts: list[np.ndarray] = []
            voltage_parts: list[np.ndarray] = []
            temperature_parts: list[np.ndarray] = []
            shift = 0.0
            for part in group:
                t_local = part["t_s"].to_numpy()
                t_parts.append(shift + t_local)
                current_parts.append(part["current_a"].to_numpy())
                voltage_parts.append(part["voltage_v"].to_numpy())
                temperature_parts.append(part["temperature_c"].to_numpy())
                shift += float(t_local[-1]) + self._sample_interval(t_local)
            t_merged = np.concatenate(t_parts)
            current = np.concatenate(current_parts)
            frames.append(
                pl.DataFrame(
                    {
                        "cell_id": [cell_id] * t_merged.size,
                        "cycle_index": np.full(t_merged.size, cycle_index, dtype=np.int32),
                        "step_type": [kind] * t_merged.size,
                        "t_s": offset + t_merged,
                        "voltage_v": np.concatenate(voltage_parts),
                        "current_a": current,
                        "temperature_c": np.concatenate(temperature_parts),
                        "charge_ah": integrate_charge(t_merged, current),
                    }
                )
            )
            offset += float(t_merged[-1]) + self._sample_interval(t_merged)
        return pl.concat(frames)

    @staticmethod
    def _sample_interval(t_s: np.ndarray) -> float:
        """Типичный интервал дискретизации, с."""
        if t_s.size < 2:
            return 0.0
        return float(np.median(np.diff(t_s)))

    def _cycle_row(self, cell_id: str, cycle_index: int, cycle: pl.DataFrame) -> dict[str, Any]:
        """Строка ``cycles``: измеренная разрядная ёмкость и признак опорного."""
        discharge = cycle.filter(pl.col("step_type") == pl.lit("dch"))
        capacity: float | None = None
        is_reference = False
        if discharge.height:
            capacity = abs(float(discharge["charge_ah"][-1]))
            rule = self.config.labels.rule(self.name)
            full_charge_v = float(rule.get("full_charge_voltage_v", 4.15))
            full_discharge_v = float(rule.get("full_discharge_voltage_v", 2.75))
            charge = cycle.filter(pl.col("step_type").is_in(["chg_cc", "chg_cv"]))
            reached_top = bool(charge.height) and (
                float(charge["voltage_v"].max()) >= full_charge_v
            )
            reached_bottom = float(discharge["voltage_v"].min()) <= full_discharge_v
            is_reference = bool(reached_top and reached_bottom and capacity > 0)
        return {
            "cell_id": cell_id,
            "cycle_index": cycle_index,
            "capacity_ah": capacity,
            "is_reference": is_reference,
        }

    def _recompute_reference(self, bundle: RawBundle, rule: Mapping[str, Any]) -> pl.DataFrame:
        """Пересчитать ``is_reference`` по границам напряжения из правила §5."""
        full_charge_v = float(rule.get("full_charge_voltage_v", 4.15))
        full_discharge_v = float(rule.get("full_discharge_voltage_v", 2.75))
        extremes = cycle_voltage_extremes(bundle.timeseries)
        flags = extremes.select(
            "cycle_index",
            (
                (pl.col("v_charge_max") >= full_charge_v)
                & (pl.col("v_discharge_min") <= full_discharge_v)
            ).alias("_full"),
        )
        return (
            bundle.cycles.drop("is_reference")
            .join(flags, on="cycle_index", how="left")
            .with_columns(pl.col("_full").fill_null(False).alias("is_reference"))
            .drop("_full")
        )

    @staticmethod
    def _drop_shallow_charges(
        cycles: pl.DataFrame, bundle: RawBundle, *, rel: float
    ) -> tuple[pl.DataFrame, int]:
        """Снять признак опорного с циклов, перенёсших слишком мало заряда.

        Правило владельца: «полный» цикл обязан не только дойти до верхней и
        нижней границы по напряжению, но и перенести существенный заряд. Иначе
        в конце жизни в опорные попадают циклы, где ячейка была почти полной:
        они дают заниженную разрядную ёмкость и ложные «восстановления» SOH.

        Порог считается от ``bol`` этой же ячейки — от ёмкости первого
        опорного цикла. Первый цикл свежей ячейки заряжается полностью, поэтому
        опорное значение берётся до фильтрации.
        """
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

    def _protocol(self, timeseries: pl.DataFrame) -> str:
        """Собрать ``charge_protocol`` из фактических токов и напряжений (§4.1)."""
        charge = timeseries.filter(pl.col("step_type").is_in(["chg_cc", "chg_cv"]))
        discharge = timeseries.filter(pl.col("step_type") == pl.lit("dch"))
        if charge.height == 0:
            return "CALCE CS2: разбор без зарядной фазы"
        cc = charge.filter(pl.col("step_type") == pl.lit("chg_cc"))
        current = float(cc["current_a"].median()) if cc.height else float("nan")
        parts = [
            f"CC {current:.2f} A ({current / NOMINAL_CAPACITY_AH:.2f}C) "
            f"до {float(charge['voltage_v'].max()):.2f} В"
        ]
        cv = charge.filter(pl.col("step_type") == pl.lit("chg_cv"))
        if cv.height:
            parts.append(f"CV до {float(cv['current_a'].min()):.2f} A")
        if discharge.height:
            parts.append(
                f"разряд CC {float(discharge['current_a'].median()):.2f} A "
                f"до {float(discharge['voltage_v'].min()):.2f} В"
            )
        return ", ".join(parts)

    @staticmethod
    def _notes(files: list[Path], notes: list[str]) -> str | None:
        """Собрать ``notes``: форматы, число файлов, аномалии разбора."""
        formats = sorted({path.suffix.lower().lstrip(".") for path in files})
        head = [
            f"разобрано файлов: {len(files)} ({', '.join(formats)})",
            "температура в логах не пишется, уставка комнатная (ТЗ §17.1)",
        ]
        if "txt" in formats:
            head.append("CADEX: Time в минутах, дискретизация 60 с")
        unique = sorted(set(notes))
        if unique:
            head.append("; ".join(unique[:5]))
        return "; ".join(head)

    # ------------------------------------------------------------------
    # mark_reference_cycles
    # ------------------------------------------------------------------

    def mark_reference_cycles(self, bundle: RawBundle) -> CellBundle:
        """Применить правило ``every_full_cycle`` и посчитать §4.2.

        Опорным считается цикл, прошедший полный заряд (до верхней границы) и
        полный разряд (до нижней). Частичные циклы типов 4-6 и циклы со
        случайной отсечкой дают несопоставимую ёмкость и в опорные не попадают.

        Признак опорного цикла пересчитывается из ``timeseries``, а не берётся
        из входного кадра: тогда правило можно применить заново к уже
        разобранным данным (``bsb labels``), не перечитывая сырые файлы.
        """
        warnings = validate_raw_bundle(bundle)
        rule = self.config.labels.rule(self.name)
        strategy = str(rule.get("strategy", "every_full_cycle"))
        if strategy != "every_full_cycle":
            raise NotImplementedError(
                f"calce: стратегия {strategy!r} не реализована (ожидается every_full_cycle)"
            )
        cycles = self._recompute_reference(bundle, rule)
        min_throughput_rel = float(rule.get("min_charge_throughput_rel", 0.0))
        if min_throughput_rel > 0:
            cycles, shallow = self._drop_shallow_charges(cycles, bundle, rel=min_throughput_rel)
            if shallow:
                warnings.append(
                    f"{bundle.cell_id}: {shallow} циклов дошли до границ по напряжению, "
                    f"но перенесли меньше {min_throughput_rel:.0%}·bol заряда "
                    "(ячейка была почти полной), в опорные не попали"
                )
        bundle = RawBundle(cells=bundle.cells, cycles=cycles, timeseries=bundle.timeseries)

        reference = bundle.cycles.filter(
            pl.col("is_reference") & pl.col("capacity_ah").is_not_null()
        )
        if reference.height == 0:
            raise SchemaViolation(f"{bundle.cell_id}: ни одного полного цикла — bol взять негде")
        skipped = bundle.cycles.height - reference.height
        if skipped:
            warnings.append(
                f"{bundle.cell_id}: {skipped} из {bundle.cycles.height} циклов неполные "
                "(частичные, с отсечкой или не тем током), в опорные не попали"
            )
        if warnings:
            existing = bundle.cells["notes"][0]
            merged = "; ".join([part for part in (existing, *warnings) if part])
            bundle = RawBundle(
                cells=bundle.cells.with_columns(pl.lit(merged).alias("notes")),
                cycles=bundle.cycles,
                timeseries=bundle.timeseries,
            )
        return build_labels(bundle, self.config)
