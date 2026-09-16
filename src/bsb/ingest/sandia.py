"""Адаптер Sandia (ТЗ §6.4, §17.4) — фаза 6 плана §15.

86 ячеек 18650 трёх химий с Battery Archive: 30 LFP, 24 NCA, 32 NMC.
Опорный цикл — capacity check (три цикла 0-100% SOC на 0.5C). Заряд у всех
ячеек 0.5C; протокол рабочего цикла зависит от режима DOD, поэтому в
бенчмарк попадают только capacity check (§5.1).

Обрабатывается тем же кодом, что и обучающие датасеты: никаких ветвлений
«если тест» в пайплайне (§6.4).
"""

from __future__ import annotations

from pathlib import Path

from bsb.config import BenchmarkConfig
from bsb.ingest.base import CellSource, RawBundle
from bsb.schema import CellBundle

__all__ = ["SandiaAdapter"]


class SandiaAdapter:
    """Адаптер Sandia / SNL с Battery Archive."""

    name = "sandia"

    def __init__(self, config: BenchmarkConfig) -> None:
        """Сохранить конфиг."""
        self.config = config

    def discover(self, raw_dir: Path) -> list[CellSource]:
        """Найти ячейки в ``raw_dir`` (фаза 6)."""
        raise NotImplementedError("SandiaAdapter.discover — фаза 6 плана §15")

    def load_cell(self, src: CellSource) -> RawBundle:
        """Прочитать одну ячейку (фаза 6)."""
        raise NotImplementedError("SandiaAdapter.load_cell — фаза 6 плана §15")

    def mark_reference_cycles(self, bundle: RawBundle) -> CellBundle:
        """Разметить capacity check как опорные циклы (фаза 6)."""
        raise NotImplementedError("SandiaAdapter.mark_reference_cycles — фаза 6 плана §15")
