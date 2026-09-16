"""Валидатор формата сабмита (ТЗ §11) — реализация в фазе 7 плана §15.

Проверки: полнота набора ``fragment_id``, диапазон ``soh_pred``,
согласованность ``soh_lo <= soh_pred <= soh_hi``, отсутствие NaN.
Сообщения обязаны указывать конкретные строки — иначе первые две недели
соревнования уйдут на переписку про форматы.
"""

from __future__ import annotations

from pathlib import Path

from bsb.config import BenchmarkConfig

__all__ = ["validate_submission"]


def validate_submission(
    submission_path: str | Path,
    *,
    expected_ids: list[str] | None = None,
    config: BenchmarkConfig | None = None,
) -> list[str]:
    """Проверить файл сабмита, вернуть список проблем (пустой — всё хорошо).

    Реализация — фаза 7 плана §15.
    """
    raise NotImplementedError("bsb.validate_submission.validate_submission — фаза 7 плана §15")
