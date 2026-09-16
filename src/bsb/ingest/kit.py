"""Адаптер KIT NMC/C-SiO (ТЗ §6.3, §17.2) — фаза 5 плана §15.

228 ячеек, 76 режимов, четыре температуры, свыше 3 млрд точек при разрешении
2 с: разбор обязан быть потоковым, датасет целиком в память не помещается.
Прореживание до ``ingest.kit.target_resolution_s`` с сохранением исходного
разрешения внутри CC-фазы заряда.

Готовые скрипты чтения: https://github.com/energystatusdata/bat-age-data-scripts/
— начинать с них, а не с нуля (§17.2).
"""

from __future__ import annotations

from pathlib import Path

from bsb.config import BenchmarkConfig
from bsb.ingest.base import CellSource
from bsb.schema import CellBundle

__all__ = ["KitAdapter"]


class KitAdapter:
    """Адаптер KIT NMC/C-SiO (потоковый разбор)."""

    name = "kit"

    def __init__(self, config: BenchmarkConfig) -> None:
        """Сохранить конфиг."""
        self.config = config

    def discover(self, raw_dir: Path) -> list[CellSource]:
        """Найти ячейки в ``raw_dir`` (фаза 5)."""
        raise NotImplementedError("KitAdapter.discover — фаза 5 плана §15")

    def load_cell(self, src: CellSource) -> CellBundle:
        """Потоково разобрать одну ячейку (фаза 5)."""
        raise NotImplementedError("KitAdapter.load_cell — фаза 5 плана §15")

    def mark_reference_cycles(self, bundle: CellBundle) -> CellBundle:
        """Разметить опорные циклы по таблице check-up (фаза 5)."""
        raise NotImplementedError("KitAdapter.mark_reference_cycles — фаза 5 плана §15")
