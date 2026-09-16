"""Опорные циклы и целевая переменная (ТЗ §5) — фаза 2 плана §15.

Разделение ответственности, зафиксированное в ТЗ §5:

* **какие циклы считать опорными** — решение владельца репозитория, по одному
  на датасет (``labels.reference_cycle_rules`` в конфиге); реализуется в
  ``DatasetAdapter.mark_reference_cycles`` и принимается глазами по графикам
  §14, а не юнит-тестом;
* **арифметика поверх опорных циклов** — ``soh = capacity_ah / bol``,
  интерполяция, ``soh_gap_cycles``, запрет экстраполяции — реализуется здесь.

Функция :func:`build_labels` реализуется в фазе 2 (владелец) и фазе 1
(минимум, нужный для сквозного прогона Oxford).
"""

from __future__ import annotations

from bsb.config import BenchmarkConfig
from bsb.schema import CellBundle

__all__ = ["build_labels"]


def build_labels(bundle: CellBundle, config: BenchmarkConfig) -> CellBundle:
    """Посчитать ``soh``, ``soh_source`` и ``soh_gap_cycles`` (ТЗ §5).

    Реализация — фаза 1/2 плана §15.
    """
    raise NotImplementedError("bsb.labels.build_labels — фаза 2 плана §15")
