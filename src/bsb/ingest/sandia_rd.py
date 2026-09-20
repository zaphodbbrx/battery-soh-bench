"""Адаптер архива циклирования R&D Data Repository Sandia (Barkholtz et al.).

Роль в бенчмарке — **только обучение**
--------------------------------------
Это другое исследование, чем SNL/Preger из §17.4 (скрытый тест): иная программа
тестирования (2016-2017, ``160920_*_Program_File.sdu``), другой набор ячеек.
Название датасета отдельное (``sandia_rd``), чтобы сплиты §8 не смешивали два
исследования.

Зачем он нужен: обучение без него состоит из LCO (CALCE CS2 и Oxford), тогда
как тест — NCA, NMC и LFP. Здесь есть все три тестовые химии (по 6 ячеек),
причём с каналами температуры, которых у CALCE нет вовсе. Это прямо закрывает
разрыв §9.

Данные
------
179 файлов ``.xls`` в 60 каталогах, 24 ячейки, температуры 5/15/25/35/45 °C.
Формат — Arbin, **тот же, что у CALCE**, поэтому чтение, разметка шагов и сборка
циклов переиспользуются из :class:`bsb.ingest.calce.CalceAdapter`. Отличия:

* шапка без паспортных границ напряжения, а химии разные (LFP до 3.6 В против
  4.2 В у NCA/NMC/LCO) — поэтому «полный» цикл определяется по данным ячейки;
* в одном файле циклы разряжаются разными токами (2.7 / 5.0 / 6.0 А), а ёмкость
  зависит от тока: опорными остаются циклы с модальным током;
* счётчик ``Discharge_Capacity(Ah)`` в исходнике не сбрасывается между шагами
  внутри цикла, поэтому ёмкость берётся из нашего интегрирования (§4.3), а
  источник используется только для сверки;
* модель ячейки и форм-фактор в данных не указаны. Форм-фактор проставлен как
  ``cylindrical_18650`` по ёмкостям (~2.4-2.9 А·ч на разряд), и это помечено в
  ``notes`` — если понадобится, владелец уточнит.

Опорные циклы
-------------
Правило ``every_full_cycle`` с тремя условиями (все — из данных ячейки):
цикл зарядился до её собственного максимума напряжения, разрядился до её
собственного минимума и перенёс не меньше половины ``bol`` заряда; ток разряда
близок к модальному для этого файла.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Final

import polars as pl

from bsb.config import BenchmarkConfig
from bsb.ingest.base import CellSource, RawBundle, charge_throughput_by_cycle
from bsb.ingest.calce import CalceAdapter
from bsb.labels import NoReferenceCycles, build_labels
from bsb.schema import CHEMISTRIES, CellBundle, SchemaViolation

__all__ = ["SandiaRdAdapter"]

#: Химия в имени файла: ``NCA_1_25C_Reg.xls``
CHEMISTRY_PATTERN: Final = re.compile(r"^(?P<chemistry>LCO|LFP|NCA|NMC)_(?P<number>\d+)")
#: Температура в имени файла: ``..._25C_...``
TEMPERATURE_PATTERN: Final = re.compile(r"_(?P<temperature>\d{1,2})C")
#: Форм-фактор в архиве не указан; проставлен по ёмкостям и помечен в notes.
INFERRED_FORM_FACTOR: Final[str] = "cylindrical_18650"


class SandiaRdAdapter(CalceAdapter):
    """Архив R&D Repository Sandia — обучение, не тест."""

    name = "sandia_rd"

    def __init__(self, config: BenchmarkConfig) -> None:
        """Сохранить конфиг (переиспользует настройки чтения CALCE)."""
        super().__init__(config)

    # ------------------------------------------------------------------
    # discover
    # ------------------------------------------------------------------

    def discover(self, raw_dir: Path) -> list[CellSource]:
        """Найти файлы архива: один ``.xls`` — одна запись «ячейка × условие».

        Физическая ячейка тестировалась при нескольких температурах, поэтому
        каждая запись идёт отдельным ``cell_id``, а номер ячейки попадает в
        ``batch_id``: §8 разрешает разбивать по партии, и это не даст одной
        физической ячейке попасть и в train, и в val.
        """
        if not raw_dir.is_dir():
            raise FileNotFoundError(f"{raw_dir}: каталог сырых данных не найден")
        files = [
            path
            for path in sorted(raw_dir.rglob("*.xls"))
            if not any(part.startswith("_") for part in path.parts)
        ]
        if not files:
            raise FileNotFoundError(
                f"{raw_dir}: нет ни одного .xls; сначала выполните scripts/download_sandia_rd.sh"
            )
        sources = []
        for path in files:
            match = CHEMISTRY_PATTERN.match(path.stem)
            if match is None:
                continue
            sources.append(
                CellSource(
                    dataset=self.name,
                    original_id=path.stem,
                    path=path,
                    extras={"files": [path], "physical_cell": match.group(0)},
                )
            )
        if not sources:
            raise SchemaViolation(f"{raw_dir}: ни один файл не опознан по имени")
        return sources

    # ------------------------------------------------------------------
    # метаданные
    # ------------------------------------------------------------------

    def _cell_dict(
        self,
        src: CellSource,
        timeseries: pl.DataFrame,
        *,
        files: list[Path],
        notes: list[str],
    ) -> dict[str, Any]:
        """Метаданные по §4.1: химия и температура из имени, форм-фактор выведен."""
        match = CHEMISTRY_PATTERN.match(src.original_id)
        chemistry = match.group("chemistry") if match else ""
        if chemistry not in CHEMISTRIES:
            raise SchemaViolation(f"{src.cell_id}: химия {chemistry!r} не в списке §4.1")
        temperature = TEMPERATURE_PATTERN.search(src.original_id)
        if temperature is None:
            raise SchemaViolation(f"{src.cell_id}: температура не найдена в имени файла")
        nominal = self._nominal_capacity(timeseries)
        return {
            "cell_id": src.cell_id,
            "dataset": self.name,
            "chemistry": chemistry,
            "form_factor": INFERRED_FORM_FACTOR,
            "nominal_capacity_ah": nominal,
            "temperature_setpoint_c": float(temperature.group("temperature")),
            "charge_protocol": self._protocol(timeseries),
            "batch_id": str(src.extras.get("physical_cell", src.original_id)),
            "notes": "; ".join(
                [
                    "источник: R&D Data Repository Sandia (Barkholtz et al. 2017); "
                    "только обучение, не тест",
                    "паспортная ёмкость и форм-фактор в архиве не указаны: "
                    "ёмкость взята по первому полному циклу, форм-фактор — по ёмкостям",
                    *sorted(set(notes))[:5],
                ]
            ),
        }

    @staticmethod
    def _nominal_capacity(timeseries: pl.DataFrame) -> float:
        """Паспортной ёмкости в архиве нет — берём наибольшую измеренную разрядную."""
        discharge = timeseries.filter(pl.col("step_type") == pl.lit("dch"))
        if discharge.height == 0:
            raise SchemaViolation("в записи нет ни одного разряда")
        capacity = (
            discharge.group_by("cycle_index")
            .agg(pl.col("charge_ah").min().abs().alias("q"))["q"]
            .max()
        )
        if capacity is None or capacity <= 0:
            raise SchemaViolation("не удалось определить ёмкость ячейки")
        return float(capacity)

    # ------------------------------------------------------------------
    # mark_reference_cycles
    # ------------------------------------------------------------------

    def mark_reference_cycles(self, bundle: RawBundle) -> CellBundle:
        """Опорные циклы по данным самой ячейки, затем §4.2 через :mod:`bsb.labels`.

        Три условия: полный размах напряжения (до собственных границ ячейки),
        модальный ток разряда и перенесённый заряд не меньше половины ``bol``.
        """
        rule = self.config.labels.rule(self.name)
        tolerance = float(rule.get("voltage_tolerance_v", 0.05))
        rate_tolerance = float(rule.get("discharge_rate_tolerance_rel", 0.02))
        min_throughput_rel = float(rule.get("min_charge_throughput_rel", 0.5))

        stats = self._cycle_stats(bundle.timeseries)
        if stats.height == 0:
            raise SchemaViolation(f"{bundle.cell_id}: нет разрядных циклов")

        top = float(stats["v_charge_max"].max())
        bottom = float(stats["v_discharge_min"].min())
        modal_rate = self._modal_rate(stats["i_discharge"])
        full = (stats["v_charge_max"] >= top - tolerance) & (
            stats["v_discharge_min"] <= bottom + tolerance
        )
        same_rate = (stats["i_discharge"] - modal_rate).abs() <= rate_tolerance * modal_rate
        shape = stats.select("cycle_index", (full & same_rate).alias("_shape"))
        cycles = bundle.cycles.join(shape, on="cycle_index", how="left").with_columns(
            pl.col("_shape").fill_null(False).alias("is_reference")
        )

        # Порог перенесённого заряда: bol берём по первому подходящему циклу,
        # он снят на свежей ячейке.
        candidates = cycles.filter(
            pl.col("is_reference") & pl.col("capacity_ah").is_not_null()
        ).sort("cycle_index")
        if candidates.height == 0:
            raise NoReferenceCycles(
                f"{bundle.cell_id}: полных циклов с модальным током разряда нет "
                "(в файле только частичные циклы)"
            )
        bol = float(candidates["capacity_ah"][0])
        throughput = charge_throughput_by_cycle(bundle.timeseries)
        joined = cycles.join(throughput, on="cycle_index", how="left").with_columns(
            pl.col("q_charge").fill_null(0.0)
        )
        dropped_shape = int((~joined["is_reference"]).sum())
        weak = joined.filter(
            pl.col("is_reference") & (pl.col("q_charge") < min_throughput_rel * bol)
        )
        cycles = joined.drop("q_charge", "_shape").with_columns(
            (pl.col("is_reference") & ~pl.col("cycle_index").is_in(weak["cycle_index"])).alias(
                "is_reference"
            )
        )

        if cycles.filter(pl.col("is_reference")).height == 0:
            raise NoReferenceCycles(
                f"{bundle.cell_id}: после фильтра по току и перенесённому заряду "
                "не осталось ни одного опорного цикла"
            )
        warnings = [
            f"{bundle.cell_id}: {dropped_shape} из {cycles.height} циклов не имеют полного "
            f"размаха при модальном токе {modal_rate:.2f} А, в опорные не попали",
        ]
        if weak.height:
            warnings.append(
                f"{bundle.cell_id}: {weak.height} циклов перенесли меньше "
                f"{min_throughput_rel:.0%}·bol заряда"
            )
        existing = bundle.cells["notes"][0]
        merged = "; ".join([part for part in (existing, *warnings) if part])
        bundle = RawBundle(
            cells=bundle.cells.with_columns(pl.lit(merged).alias("notes")),
            cycles=cycles,
            timeseries=bundle.timeseries,
        )
        return build_labels(bundle, self.config)

    @staticmethod
    def _cycle_stats(timeseries: pl.DataFrame) -> pl.DataFrame:
        """По циклу: границы напряжения, ток разряда и зарядная ёмкость."""
        charge = (
            timeseries.filter(pl.col("step_type").is_in(["chg_cc", "chg_cv"]))
            .group_by("cycle_index")
            .agg(pl.col("voltage_v").max().alias("v_charge_max"))
        )
        discharge = (
            timeseries.filter(pl.col("step_type") == pl.lit("dch"))
            .group_by("cycle_index")
            .agg(
                pl.col("voltage_v").min().alias("v_discharge_min"),
                pl.col("current_a").median().abs().alias("i_discharge"),
            )
        )
        return charge.join(discharge, on="cycle_index", how="inner")

    @staticmethod
    def _modal_rate(current: pl.Series) -> float:
        """Модальный ток разряда: в одном файле циклы идут на разных токах."""
        counts = current.round(2).alias("value").value_counts(sort=True)
        if counts.height == 0:
            raise SchemaViolation("не удалось определить модальный ток разряда")
        return float(counts["value"][0])
