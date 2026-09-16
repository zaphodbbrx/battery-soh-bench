"""Адаптер CALCE (ТЗ §6.2, §17.1) — фаза 4 плана §15.

Серия CS2: 15 призматических LCO 1.1 А·ч, заряд единый CC-CV 0.5C до 4.2 В
с отсечкой 0.05 А. Формат — Excel-файлы по датасетам; CS2_8 и CS2_21
тестировались на CADEX и лежат в ``.txt``.

Решение по остальным сериям — открытый вопрос §18 п.5; текущее значение
``ingest.calce.series`` в конфиге.
"""

from __future__ import annotations

from pathlib import Path

from bsb.config import BenchmarkConfig
from bsb.ingest.base import CellSource, RawBundle
from bsb.schema import CellBundle

__all__ = ["CalceAdapter"]


class CalceAdapter:
    """Адаптер CALCE (серии из ``ingest.calce.series``)."""

    name = "calce"

    def __init__(self, config: BenchmarkConfig) -> None:
        """Сохранить конфиг."""
        self.config = config

    def discover(self, raw_dir: Path) -> list[CellSource]:
        """Найти ячейки в ``raw_dir`` (фаза 4)."""
        raise NotImplementedError("CalceAdapter.discover — фаза 4 плана §15")

    def load_cell(self, src: CellSource) -> RawBundle:
        """Прочитать одну ячейку (фаза 4)."""
        raise NotImplementedError("CalceAdapter.load_cell — фаза 4 плана §15")

    def mark_reference_cycles(self, bundle: RawBundle) -> CellBundle:
        """Разметить опорные циклы по правилу ``every_full_cycle`` (фаза 4)."""
        raise NotImplementedError("CalceAdapter.mark_reference_cycles — фаза 4 плана §15")
