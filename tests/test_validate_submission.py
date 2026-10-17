"""Тесты валидатора сабмита (ТЗ §11, §14).

ТЗ §14 требует, чтобы валидатор ловил каждый из четырёх классов ошибок:
полнота, диапазон, согласованность интервалов, NaN.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from bsb.config import BenchmarkConfig
from bsb.validate_submission import (
    expected_test_ids,
    read_submission,
    summarize_predictions,
    validate_submission,
)


def _frame(rows: list[dict[str, Any]]) -> pl.DataFrame:
    """Кадр сабмита с правильными типами."""
    return pl.DataFrame(
        rows,
        schema={
            "fragment_id": pl.String,
            "soh_pred": pl.Float64,
            "soh_lo": pl.Float64,
            "soh_hi": pl.Float64,
        },
    )


@pytest.fixture
def valid_rows() -> list[dict[str, Any]]:
    """Три корректные строки сабмита с интервалами."""
    return [
        {"fragment_id": "a", "soh_pred": 0.95, "soh_lo": 0.90, "soh_hi": 1.0},
        {"fragment_id": "b", "soh_pred": 0.85, "soh_lo": 0.80, "soh_hi": 0.90},
        {"fragment_id": "c", "soh_pred": 0.75, "soh_lo": 0.70, "soh_hi": 0.80},
    ]


def test_valid_submission_passes(valid_rows: list[dict[str, Any]]) -> None:
    """Корректный сабмит проходит и участвует в калибровке."""
    report = validate_submission(_frame(valid_rows), expected_ids=["a", "b", "c"])
    assert report.ok, report.errors
    assert report.has_intervals
    assert report.n_rows == 3


def test_missing_and_extra_ids_are_reported(valid_rows: list[dict[str, Any]]) -> None:
    """Полнота: не хватает одного, лишний один — обе ошибки названы."""
    report = validate_submission(_frame(valid_rows), expected_ids=["a", "b", "d"])
    assert not report.ok
    assert any("не хватает 1" in item and "'d'" in item for item in report.errors)
    assert any("лишние 1" in item and "'c'" in item for item in report.errors)


def test_out_of_range_prediction_is_reported(valid_rows: list[dict[str, Any]]) -> None:
    """Диапазон: (0, 1.2] — границы взяты из конфига."""
    rows = [dict(row) for row in valid_rows]
    rows[1]["soh_pred"] = 1.4
    rows[2]["soh_pred"] = 0.0
    report = validate_submission(_frame(rows), expected_ids=["a", "b", "c"])
    assert not report.ok
    assert any("вне (0.0, 1.2]" in item for item in report.errors)


def test_inconsistent_intervals_are_reported(valid_rows: list[dict[str, Any]]) -> None:
    """Согласованность: soh_lo <= soh_pred <= soh_hi."""
    rows = [dict(row) for row in valid_rows]
    rows[0]["soh_lo"] = 0.99
    report = validate_submission(_frame(rows), expected_ids=["a", "b", "c"])
    assert not report.ok
    assert any("soh_lo <= soh_pred <= soh_hi" in item for item in report.errors)
    assert any("'a'" in item for item in report.errors)


def test_nan_prediction_is_reported(valid_rows: list[dict[str, Any]]) -> None:
    """NaN в soh_pred или в границах интервала — ошибка."""
    rows = [dict(row) for row in valid_rows]
    rows[2]["soh_pred"] = float("nan")
    report = validate_submission(_frame(rows), expected_ids=["a", "b", "c"])
    assert not report.ok
    assert any("NaN" in item for item in report.errors)


def test_null_bounds_are_reported(valid_rows: list[dict[str, Any]]) -> None:
    """Пропуск в границах интервала — тоже ошибка, а не тихое «нет калибровки»."""
    rows = [dict(row) for row in valid_rows]
    rows[1]["soh_hi"] = None
    report = validate_submission(_frame(rows), expected_ids=["a", "b", "c"])
    assert not report.ok
    assert any("soh_hi пуст" in item for item in report.errors)


def test_partial_interval_columns_are_rejected(valid_rows: list[dict[str, Any]]) -> None:
    """Одна колонка интервала без второй — ошибка формата."""
    frame = _frame(valid_rows).drop("soh_hi")
    report = validate_submission(frame, expected_ids=["a", "b", "c"])
    assert not report.ok
    assert any("неполностью" in item for item in report.errors)


def test_duplicate_ids_are_reported(valid_rows: list[dict[str, Any]]) -> None:
    """Дубли fragment_id ломают сопоставление с эталоном."""
    report = validate_submission(_frame([*valid_rows, valid_rows[0]]))
    assert not report.ok
    assert any("повторяющиеся" in item for item in report.errors)


def test_missing_columns_are_reported() -> None:
    """Без обязательных колонок сабмит отвергается сразу."""
    frame = pl.DataFrame({"id": ["a"], "prediction": [0.9]})
    report = validate_submission(frame)
    assert not report.ok
    assert any("нет обязательных колонок" in item for item in report.errors)


def test_empty_submission_is_reported() -> None:
    """Пустой сабмит — ошибка, а не «ноль строк прошло»."""
    frame = pl.DataFrame(schema={"fragment_id": pl.String, "soh_pred": pl.Float64})
    report = validate_submission(frame)
    assert not report.ok
    assert any("пуст" in item for item in report.errors)


def test_without_intervals_calibration_is_unavailable(valid_rows: list[dict[str, Any]]) -> None:
    """§11: без интервалов команда не участвует в номинации «калибровка»."""
    frame = _frame(valid_rows).drop("soh_lo", "soh_hi")
    report = validate_submission(frame, expected_ids=["a", "b", "c"])
    assert report.ok
    assert not report.has_intervals
    assert any("калибровка" in item for item in report.warnings)


def test_csv_roundtrip_and_line_numbers(tmp_path: Path, valid_rows: list[dict[str, Any]]) -> None:
    """CSV читается, а номера строк в сообщениях учитывают заголовок."""
    path = tmp_path / "submission.csv"
    _frame(valid_rows).write_csv(path)
    rows = [dict(row) for row in valid_rows]
    rows[1]["soh_pred"] = 5.0
    _frame(rows).write_csv(path)
    report = validate_submission(path, expected_ids=["a", "b", "c"])
    assert not report.ok
    # вторая строка данных = третья строка файла
    assert any("строк[ах]* [3]" in item or "[3]" in item for item in report.errors)


def test_parquet_is_supported(tmp_path: Path, valid_rows: list[dict[str, Any]]) -> None:
    """Parquet — такой же допустимый формат, как CSV (ТЗ §11)."""
    path = tmp_path / "submission.parquet"
    _frame(valid_rows).write_parquet(path)
    assert read_submission(path).height == 3
    assert validate_submission(path, expected_ids=["a", "b", "c"]).ok


def test_all_empty_interval_columns_do_not_crash(
    tmp_path: Path, valid_rows: list[dict[str, Any]]
) -> None:
    """CSV с целиком пустыми колонками интервалов читается как строка.

    Найдено смоук-тестом: без приведения типов проверка ``is_finite`` падала
    с InvalidOperationError вместо внятного сообщения.
    """
    rows = [{**row, "soh_lo": None, "soh_hi": None} for row in valid_rows]
    path = tmp_path / "empty_bounds.csv"
    _frame(rows).write_csv(path)
    report = validate_submission(path, expected_ids=["a", "b", "c"])
    assert not report.ok
    assert any("soh_lo пуст" in item for item in report.errors)
    assert any("soh_hi пуст" in item for item in report.errors)


def test_non_numeric_values_are_reported(tmp_path: Path) -> None:
    """Текст в числовой колонке сообщается как «не число», а не как падение."""
    path = tmp_path / "garbage.csv"
    path.write_text(
        "fragment_id,soh_pred,soh_lo,soh_hi\na,0.9,0.8,1.0\nb,not-a-number,0.8,1.0\n",
        encoding="utf-8",
    )
    report = validate_submission(path, expected_ids=["a", "b"])
    assert not report.ok
    assert any("не являются числами" in item for item in report.errors)
    assert any("soh_pred пуст" in item for item in report.errors)


def test_missing_file_is_reported(tmp_path: Path) -> None:
    """Отсутствующий файл — внятная ошибка."""
    with pytest.raises(FileNotFoundError):
        read_submission(tmp_path / "nope.csv")


def test_unsupported_extension_is_rejected(tmp_path: Path) -> None:
    """Расширение не из §11 — ошибка формата."""
    path = tmp_path / "submission.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="CSV или parquet"):
        read_submission(path)


def test_expected_ids_come_from_config(
    make_config: Callable[[dict[str, Any] | None], BenchmarkConfig],
    tmp_path: Path,
) -> None:
    """ID тестовых фрагментов читаются из открытого файла без меток."""
    config = make_config()
    assert expected_test_ids(config) is None, "пока тест не сгенерирован"
    path = config.paths.fragments_path("test", labeled=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "fragment_id": ["x", "y"],
            "cell_id": ["sandia_Cell1"] * 2,
            "cycle_index": [1, 2],
            "v_start": [3.9, 3.9],
            "v_end": [4.1, 4.1],
            "q_norm": [[0.1, 0.2], [0.1, 0.2]],
            "current": [[0.5, 0.5], [0.5, 0.5]],
            "temperature": [[25.0, 25.0], [25.0, 25.0]],
            "soh": [1.0, 0.99],
        }
    ).write_parquet(path)
    assert expected_test_ids(config) == ["x", "y"]


def test_summarize_predictions_flags_bound_saturation() -> None:
    """Сводка показывает, не упёрлась ли модель в границы диапазона."""
    frame = pl.DataFrame({"soh_pred": [0.0, 0.5, 1.2, 1.2]})
    summary = summarize_predictions(frame)
    assert summary["n"] == 4
    assert summary["at_lower_bound"] == pytest.approx(0.25)
    assert summary["at_upper_bound"] == pytest.approx(0.5)
