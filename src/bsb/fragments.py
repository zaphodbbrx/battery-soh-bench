"""Генератор фрагментов (ТЗ §7) — единственная точка порождения входов модели.

Реализация — фаза 3 плана §15. Параметры берутся только из
``config.fragment``; значение по умолчанию ``v_normalization: per_cell_range``
обосновано в §9 (иначе на LFP окна не существуют физически).
"""

from __future__ import annotations

from bsb.config import BenchmarkConfig
from bsb.schema import CellBundle

__all__ = ["generate_fragments"]


def generate_fragments(bundle: CellBundle, config: BenchmarkConfig) -> list[dict[str, object]]:
    """Построить фрагменты одной ячейки в режиме ``fixed`` (ТЗ §7).

    Реализация — фаза 3 плана §15.
    """
    raise NotImplementedError("bsb.fragments.generate_fragments — фаза 3 плана §15")
