"""Валидатор формата сабмита (ТЗ §11).

Проверяет:

* **полноту** — все ``fragment_id`` теста присутствуют, лишних нет;
* **диапазон** — ``soh_pred`` в ``(0, 1.2]``;
* **согласованность** — ``soh_lo <= soh_pred <= soh_hi``;
* **отсутствие NaN**.

Сообщения указывают конкретные строки файла: без этого первые две недели
соревнования уйдут на переписку про форматы.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from bsb.config import BenchmarkConfig

__all__ = [
    "NUMERIC_COLUMNS",
    "REQUIRED_COLUMNS",
    "SubmissionReport",
    "coerce_numeric_columns",
    "read_submission",
    "validate_submission",
]

#: Обязательные колонки сабмита; ``soh_lo`` / ``soh_hi`` опциональны (ТЗ §11).
REQUIRED_COLUMNS: tuple[str, ...] = ("fragment_id", "soh_pred")
INTERVAL_COLUMNS: tuple[str, ...] = ("soh_lo", "soh_hi")
#: Колонки, которые обязаны быть числами.
NUMERIC_COLUMNS: tuple[str, ...] = ("soh_pred", "soh_lo", "soh_hi")
MAX_REPORTED = 10


@dataclass(frozen=True, slots=True)
class SubmissionReport:
    """Результат проверки сабмита."""

    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    n_rows: int = 0
    has_intervals: bool = False

    @property
    def ok(self) -> bool:
        """Сабмит прошёл проверку, если нет ни одной ошибки."""
        return not self.errors

    def render(self) -> str:
        """Человекочитаемый отчёт для stdout."""
        lines = [f"строк: {self.n_rows}", f"интервалы: {'да' if self.has_intervals else 'нет'}"]
        lines.extend(f"ошибка: {item}" for item in self.errors)
        lines.extend(f"предупреждение: {item}" for item in self.warnings)
        lines.append(
            "сабмит корректен" if self.ok else f"сабмит отвергнут: {len(self.errors)} ошибок"
        )
        return "\n".join(lines)


def coerce_numeric_columns(frame: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, int]]:
    """Привести числовые колонки сабмита к ``Float64``.

    CSV с полностью пустой колонкой ``soh_lo``/``soh_hi`` читается как строка,
    и без этого шага любой арифметический анализ падает на типе. Возвращает
    кадр и число значений по колонкам, которые не удалось разобрать как число
    (они становятся пропусками и далее репортуются как ошибки).
    """
    lost: dict[str, int] = {}
    for column in NUMERIC_COLUMNS:
        if column not in frame.columns:
            continue
        series = frame[column]
        if series.dtype.is_numeric():
            frame = frame.with_columns(pl.col(column).cast(pl.Float64))
            continue
        text = series.cast(pl.String)
        present = text.is_not_null() & (text.str.strip_chars() != "")
        frame = frame.with_columns(text.cast(pl.Float64, strict=False).alias(column))
        count = int((present & frame[column].is_null()).sum())
        if count:
            lost[column] = count
    return frame, lost


def _read_raw(path: str | Path) -> pl.DataFrame:
    """Прочитать сабмит как есть, без приведения типов."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"сабмит не найден: {file_path}")
    if file_path.suffix.lower() in {".parquet", ".pq"}:
        return pl.read_parquet(file_path)
    if file_path.suffix.lower() in {".csv", ".txt"}:
        return pl.read_csv(file_path)
    raise ValueError(f"{file_path}: ожидался CSV или parquet")


def read_submission(path: str | Path) -> pl.DataFrame:
    """Прочитать сабмит в CSV или parquet (ТЗ §11) с приведением чисел."""
    frame, _lost = coerce_numeric_columns(_read_raw(path))
    return frame


def _line_numbers(
    mask: pl.Series, *, offset: int, limit: int = MAX_REPORTED
) -> tuple[list[int], int]:
    """Номера строк файла, где маска истинна (offset = 2 для CSV с заголовком)."""
    positions = mask.arg_true().to_list()
    lines = [int(position) + offset for position in positions[:limit]]
    return lines, len(positions)


def _sample_ids(frame: pl.DataFrame, mask: pl.Series, *, limit: int = MAX_REPORTED) -> str:
    """Первые ``limit`` значений ``fragment_id`` из отобранных строк."""
    values = frame.filter(mask)["fragment_id"].head(limit).to_list()
    return ", ".join(repr(value) for value in values)


def validate_submission(
    submission: str | Path | pl.DataFrame,
    *,
    expected_ids: list[str] | None = None,
    config: BenchmarkConfig | None = None,
) -> SubmissionReport:
    """Проверить файл (или кадр) сабмита.

    Parameters
    ----------
    submission
        Путь к CSV/parquet либо уже прочитанный кадр.
    expected_ids
        Полный список ``fragment_id`` теста. Если не задан, берётся из
        ``config.paths.fragments_path("test", labeled=False)`` (публичный тестовый файл).
    config
        Конфиг: границы предсказания (``eval.prediction_bounds``) и путь к
        тестовым фрагментам.

    Returns
    -------
    SubmissionReport
        Ошибки, предупреждения и признак пригодности сабмита.
    """
    errors: list[str] = []
    warnings: list[str] = []
    offset = 1
    if isinstance(submission, pl.DataFrame):
        frame = submission
    else:
        path = Path(submission)
        # Читаем сырой текст, чтобы сообщить о значениях, которые не числа,
        # а не молча превратить их в пропуски.
        frame = _read_raw(path)
        offset = 2 if path.suffix.lower() in {".csv", ".txt"} else 1

    frame, lost_values = coerce_numeric_columns(frame)
    for column, count in lost_values.items():
        errors.append(f"{column}: {count} значений не являются числами и считаются пропусками")

    bounds = config.eval.prediction_bounds if config is not None else (0.0, 1.2)
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing_columns:
        return SubmissionReport(
            errors=(f"нет обязательных колонок {missing_columns}; ожидались {REQUIRED_COLUMNS}",),
            n_rows=frame.height,
        )
    present_intervals = [column for column in INTERVAL_COLUMNS if column in frame.columns]
    has_intervals = len(present_intervals) == len(INTERVAL_COLUMNS)
    if present_intervals and not has_intervals:
        errors.append(
            f"интервалы заданы неполностью: есть {present_intervals}, "
            f"нужны обе колонки {list(INTERVAL_COLUMNS)} или ни одной"
        )

    unknown_extra = [
        column for column in frame.columns if column not in (*REQUIRED_COLUMNS, *INTERVAL_COLUMNS)
    ]
    if unknown_extra:
        warnings.append(f"лишние колонки игнорируются: {unknown_extra}")

    if frame.height == 0:
        errors.append("сабмит пуст: ни одной строки")
        return SubmissionReport(errors=tuple(errors), warnings=tuple(warnings), n_rows=0)

    null_id = frame["fragment_id"].is_null()
    if null_id.any():
        lines, count = _line_numbers(null_id, offset=offset)
        errors.append(f"пустой fragment_id в строках {lines} (всего {count})")
    duplicate = frame["fragment_id"].is_duplicated()
    if duplicate.any():
        lines, count = _line_numbers(duplicate, offset=offset)
        errors.append(f"повторяющиеся fragment_id в строках {lines} (всего {count})")

    prediction = frame["soh_pred"]
    non_numeric = prediction.is_null() | ~prediction.is_finite()
    if non_numeric.any():
        lines, count = _line_numbers(non_numeric, offset=offset)
        errors.append(f"soh_pred пуст или NaN в строках {lines} (всего {count})")
    else:
        low, high = bounds
        out_of_range = (prediction <= low) | (prediction > high)
        if out_of_range.any():
            lines, count = _line_numbers(out_of_range, offset=offset)
            examples = prediction.filter(out_of_range).head(MAX_REPORTED).to_list()
            errors.append(
                f"soh_pred вне ({low}, {high}] в строках {lines} (всего {count}); "
                f"примеры значений: {examples}"
            )

    if has_intervals:
        for column in INTERVAL_COLUMNS:
            broken = frame[column].is_null() | ~frame[column].is_finite()
            if broken.any():
                lines, count = _line_numbers(broken, offset=offset)
                errors.append(f"{column} пуст или NaN в строках {lines} (всего {count})")
        if not errors or all("пуст или NaN" not in item for item in errors):
            inconsistent = (frame["soh_lo"] > frame["soh_pred"]) | (
                frame["soh_pred"] > frame["soh_hi"]
            )
            if inconsistent.any():
                lines, count = _line_numbers(inconsistent, offset=offset)
                errors.append(
                    f"нарушено soh_lo <= soh_pred <= soh_hi в строках {lines} (всего {count}): "
                    f"{_sample_ids(frame, inconsistent)}"
                )
        empty_width = frame["soh_hi"] <= frame["soh_lo"]
        if empty_width.any():
            lines, count = _line_numbers(empty_width, offset=offset)
            errors.append(
                f"нулевая или отрицательная ширина интервала в строках {lines} (всего {count})"
            )
    else:
        warnings.append(
            "интервалы не заданы: команда не участвует в номинации «калибровка» (ТЗ §11)"
        )

    if expected_ids is not None:
        expected = set(expected_ids)
        provided = set(frame["fragment_id"].drop_nulls().to_list())
        missing = sorted(expected - provided)
        extra = sorted(provided - expected)
        if missing:
            head = ", ".join(repr(item) for item in missing[:MAX_REPORTED])
            errors.append(f"не хватает {len(missing)} fragment_id теста, например: {head}")
        if extra:
            head = ", ".join(repr(item) for item in extra[:MAX_REPORTED])
            errors.append(f"лишние {len(extra)} fragment_id, которых нет в тесте, например: {head}")
        if not missing and not extra:
            warnings.append(f"полнота: все {len(expected)} fragment_id теста на месте")
    elif config is not None:
        warnings.append(
            "список fragment_id теста не передан и не найден в "
            f"{config.paths.fragments_path('test', labeled=False)}: полнота не проверена"
        )
    return SubmissionReport(
        errors=tuple(errors),
        warnings=tuple(warnings),
        n_rows=frame.height,
        has_intervals=has_intervals,
    )


def expected_test_ids(config: BenchmarkConfig) -> list[str] | None:
    """Идентификаторы тестовых фрагментов из публичного файла без меток (§7)."""
    path = config.paths.fragments_path("test", labeled=False)
    if not path.is_file():
        return None
    return pl.read_parquet(path, columns=["fragment_id"])["fragment_id"].to_list()


def summarize_predictions(
    frame: pl.DataFrame, *, bounds: tuple[float, float] = (0.0, 1.2)
) -> dict[str, float]:
    """Краткая сводка распределения предсказаний (для протокола прогона)."""
    values = frame["soh_pred"].drop_nulls().to_numpy()
    if values.size == 0:
        return {"n": 0.0}
    low, high = bounds
    return {
        "n": float(values.size),
        "mean": float(np.mean(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "at_lower_bound": float(np.mean(values <= low)),
        "at_upper_bound": float(np.mean(values >= high)),
    }
