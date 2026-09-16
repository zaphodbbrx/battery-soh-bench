"""Бейзлайны (ТЗ §12) — фазы 8 и 8b плана §15.

Четыре публикуемых числа: «пол» (``constant``), «проба на утечку» (``probe``),
«бейзлайн» (``elastic_net``) и «потолок» (:mod:`bsb.noise_floor`, §12.4).
"""

from __future__ import annotations

from typing import Literal

import polars as pl

from bsb.config import BenchmarkConfig

__all__ = ["MODEL_NAMES", "fit_predict"]

#: Доступные модели. ``probe`` намеренно не получает саму кривую (ТЗ §12.3).
MODEL_NAMES: tuple[str, ...] = ("constant", "probe", "elastic_net")

ModelName = Literal["constant", "probe", "elastic_net"]


def fit_predict(
    train: pl.DataFrame,
    test: pl.DataFrame,
    *,
    model: str,
    config: BenchmarkConfig,
) -> pl.DataFrame:
    """Обучить модель на ``train`` и предсказать ``test`` (фаза 8).

    Возвращает кадр с ``fragment_id``, ``soh_pred`` и, где применимо,
    ``soh_lo`` / ``soh_hi``.
    """
    raise NotImplementedError(f"bsb.baseline.fit_predict({model!r}) — фаза 8 плана §15")
