"""Адаптер Oxford (ТЗ §6.1) — первый по плану §15.

8 pouch-ячеек LCO, камера 40 °C, заряд CC-CV, разряд по ездовому профилю
Artemis, характеризация каждые 100 циклов.

Ключевой факт (§17.3): в наборе лежат **только характеризационные циклы**.
``OxfordBatteryDegradationDataset_1.mat`` содержит заряд-разряд на C/18.5
(40 мА), снятый каждые 100 рабочих циклов; ``ExampleDCC1.mat`` — один первый
цикл целиком. Рабочих кривых в наборе нет, поэтому опорный цикл здесь — сам
цикл характеризации, а ``soh_gap_cycles`` всегда 0 (режим §5.1).
"""

from __future__ import annotations

from pathlib import Path

from bsb.config import BenchmarkConfig
from bsb.ingest.base import CellSource
from bsb.schema import CellBundle

__all__ = ["OxfordAdapter"]


class OxfordAdapter:
    """Адаптер Oxford Battery Degradation Dataset.

    Реализация — фаза 1 плана §15. До неё методы поднимают
    :class:`NotImplementedError`, чтобы пайплайн нельзя было случайно
    прогнать на нереализованном адаптере и получить пустой результат.
    """

    name = "oxford"

    def __init__(self, config: BenchmarkConfig) -> None:
        """Сохранить конфиг (пороги §4.4 и правила §5 берутся только отсюда)."""
        self.config = config

    def discover(self, raw_dir: Path) -> list[CellSource]:
        """Найти ячейки в ``raw_dir`` (фаза 1)."""
        raise NotImplementedError("OxfordAdapter.discover — фаза 1 плана §15")

    def load_cell(self, src: CellSource) -> CellBundle:
        """Прочитать одну ячейку из ``.mat`` (фаза 1)."""
        raise NotImplementedError("OxfordAdapter.load_cell — фаза 1 плана §15")

    def mark_reference_cycles(self, bundle: CellBundle) -> CellBundle:
        """Разметить опорные циклы (в Oxford они и есть все циклы набора)."""
        raise NotImplementedError("OxfordAdapter.mark_reference_cycles — фаза 1 плана §15")
