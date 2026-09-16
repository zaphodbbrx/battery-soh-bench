"""Тесты сплитов и проверки утечки (ТЗ §8, §14)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from bsb.config import BenchmarkConfig
from bsb.splits import (
    CellLeakageError,
    assert_no_cell_leakage,
    load_splits,
    read_split,
    split_path,
    write_split,
)


def test_write_and_read_split_roundtrip(tmp_path: Path) -> None:
    """Список читается обратно, комментарии и пустые строки игнорируются."""
    path = tmp_path / "train_cells.txt"
    write_split(path, ["oxford_Cell3", "oxford_Cell1"], header="тестовый сплит")
    assert read_split(path) == ["oxford_Cell1", "oxford_Cell3"]
    assert read_split(path) == sorted(read_split(path)), "файл записан отсортированным"


def test_read_split_rejects_duplicates(tmp_path: Path) -> None:
    """Дубль внутри файла — ошибка с номером строки."""
    path = tmp_path / "dup.txt"
    path.write_text("oxford_Cell1\noxford_Cell1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="дубликаты"):
        read_split(path)


def test_read_split_rejects_missing_file(tmp_path: Path) -> None:
    """Отсутствующий файл сплита не превращается в пустой список."""
    with pytest.raises(FileNotFoundError):
        read_split(tmp_path / "missing.txt")


def test_read_split_rejects_whitespace_inside_id(tmp_path: Path) -> None:
    """Пробел внутри идентификатора — почти наверняка склейка строк."""
    path = tmp_path / "bad.txt"
    path.write_text("oxford_Cell1 oxford_Cell2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="пробелов"):
        read_split(path)


def test_load_splits_uses_config_filenames(
    make_config: Callable[[dict[str, Any] | None], BenchmarkConfig],
) -> None:
    """Имена файлов берутся из конфига, пути — от корня репозитория."""
    config = make_config()
    write_split(split_path(config, "train"), ["oxford_Cell1"])
    write_split(split_path(config, "val"), ["kit_Cell1"])
    write_split(split_path(config, "test"), ["sandia_Cell1"])
    splits = load_splits(config)
    assert splits == {"train": ["oxford_Cell1"], "val": ["kit_Cell1"], "test": ["sandia_Cell1"]}


def test_no_leakage_on_disjoint_splits() -> None:
    """Непересекающиеся сплиты проходят проверку."""
    assert_no_cell_leakage({"train": ["a_1", "a_2"], "val": ["b_1"], "test": ["c_1"]})


def test_leakage_is_detected_with_names() -> None:
    """Пересечение ячеек ловится с указанием пары сплитов."""
    with pytest.raises(CellLeakageError, match="утечка ячеек между 'train' и 'val'"):
        assert_no_cell_leakage({"train": ["a_1"], "val": ["a_1"], "test": []})


def test_duplicates_inside_split_are_detected() -> None:
    """Дубль внутри одного сплита — тоже утечка."""
    with pytest.raises(CellLeakageError, match="дубликаты внутри сплита"):
        assert_no_cell_leakage({"train": ["a_1", "a_1"], "val": [], "test": []})


def test_batch_leakage_is_detected() -> None:
    """§8: при известной партии единица разбиения — партия."""
    splits = {"train": ["a_1"], "val": ["a_2"], "test": []}
    assert_no_cell_leakage(splits)  # по ячейкам пересечения нет
    with pytest.raises(CellLeakageError, match="утечка партий"):
        assert_no_cell_leakage(splits, cell_to_batch={"a_1": "batch7", "a_2": "batch7"})


def test_batch_check_passes_for_pure_batches() -> None:
    """Партии внутри одного сплита не считаются утечкой."""
    assert_no_cell_leakage(
        {"train": ["a_1", "a_2"], "val": ["b_1"], "test": []},
        cell_to_batch={"a_1": "batch7", "a_2": "batch7", "b_1": "batch8"},
    )
