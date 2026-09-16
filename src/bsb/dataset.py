"""Участнический доступ к фрагментам (ТЗ §7), включая режим ``sampling``.

Реализация — фаза 3 плана §15.
"""

from __future__ import annotations

from bsb.config import BenchmarkConfig

__all__ = ["load_split_fragments"]


def load_split_fragments(split: str, config: BenchmarkConfig) -> object:
    """Загрузить фрагменты сплита для обучения (режим ``fixed``/``sampling``)."""
    raise NotImplementedError("bsb.dataset.load_split_fragments — фаза 3 плана §15")
