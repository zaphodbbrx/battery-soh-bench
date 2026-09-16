"""Признаки окна для бейзлайна (ТЗ §12.5) — фаза 8 плана §15.

Обязательные признаки: ``dQ/dV`` со сглаживанием Савицкого-Голея, статистики
``dQ/dV`` внутри окна, ``v_start`` / ``v_end`` / ширина, средний и максимальный
ток (иначе модель спутает сдвиг по току со старением, §17.3), средняя
температура, интеграл ``q_norm``.
"""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl

from bsb.config import BenchmarkConfig

__all__ = ["window_features"]


def window_features(
    fragments: pl.DataFrame,
    config: BenchmarkConfig,
    *,
    columns: Sequence[str] | None = None,
) -> pl.DataFrame:
    """Построить таблицу признаков по фрагментам (фаза 8)."""
    raise NotImplementedError("bsb.features.window_features — фаза 8 плана §15")
