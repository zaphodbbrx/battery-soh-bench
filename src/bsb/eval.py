"""Метрики и разбивки (ТЗ §10).

Основные: MAE в процентных пунктах SOH, P95 абсолютной ошибки, разбивка по
химии, температуре и децилям SOH. Для номинации «калибровка» — покрытие 90%
интервала и средняя ширина. Реализация — фаза 7 плана §15.
"""

from __future__ import annotations

import polars as pl

from bsb.config import BenchmarkConfig

__all__ = ["evaluate"]


def evaluate(
    submission: pl.DataFrame,
    truth: pl.DataFrame,
    *,
    config: BenchmarkConfig,
    hard_mode: bool = False,
) -> dict[str, object]:
    """Посчитать метрики сабмита против эталона (фаза 7).

    ``hard_mode=True`` — бонусная номинация LFP (ТЗ §9).
    """
    raise NotImplementedError("bsb.eval.evaluate — фаза 7 плана §15")
