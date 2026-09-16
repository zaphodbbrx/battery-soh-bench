"""Сплиты по ячейкам (ТЗ §8).

Единица разбиения — ячейка, а при известной партии — партия. Списки лежат в
``splits/*.txt``, коммитятся и генерируются один раз. Функция
:func:`assert_no_cell_leakage` вызывается в CI и падает при пересечении.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from bsb.config import BenchmarkConfig

__all__ = [
    "CellLeakageError",
    "assert_no_cell_leakage",
    "load_splits",
    "read_split",
    "split_path",
    "write_split",
]


class CellLeakageError(ValueError):
    """Ячейка (или партия) встречается более чем в одном сплите."""


def split_path(config: BenchmarkConfig, split: str) -> Path:
    """Путь к файлу со списком ячеек сплита."""
    return config.paths.splits_dir / config.splits.filename(split)


def read_split(path: str | Path) -> list[str]:
    """Прочитать текстовый список ``cell_id``.

    Пустые строки и строки, начинающиеся с ``#``, игнорируются. Порядок
    сохраняется, дубликаты внутри файла — ошибка.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"файл сплита не найден: {file_path}")
    identifiers: list[str] = []
    for lineno, raw in enumerate(file_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if any(char.isspace() for char in line):
            raise ValueError(
                f"{file_path}:{lineno}: cell_id не должен содержать пробелов: {line!r}"
            )
        identifiers.append(line)
    duplicates = sorted({item for item in identifiers if identifiers.count(item) > 1})
    if duplicates:
        raise ValueError(f"{file_path}: дубликаты cell_id: {duplicates}")
    return identifiers


def write_split(path: str | Path, identifiers: Iterable[str], *, header: str | None = None) -> None:
    """Записать список ``cell_id`` в текстовый файл (отсортированным)."""
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    if header:
        lines.extend(f"# {line}" for line in header.splitlines())
    lines.extend(sorted(identifiers))
    file_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_splits(config: BenchmarkConfig) -> dict[str, list[str]]:
    """Прочитать все сплиты, перечисленные в конфиге (ТЗ §8)."""
    return {split: read_split(split_path(config, split)) for split in config.splits.split_names}


def assert_no_cell_leakage(
    splits: Mapping[str, Sequence[str]],
    *,
    cell_to_batch: Mapping[str, str] | None = None,
) -> None:
    """Упасть, если ячейки или партии пересекаются между сплитами (ТЗ §8).

    Parameters
    ----------
    splits
        Отображение «имя сплита → список ``cell_id``».
    cell_to_batch
        Необязательное отображение ``cell_id → batch_id``. Если задано,
        дополнительно проверяется, что ни одна партия не размазана по сплитам
        (единица разбиения — партия, ТЗ §8).

    Raises
    ------
    CellLeakageError
        При пересечении ячеек или партий между любыми двумя сплитами.
    """
    names = list(splits)
    for name in names:
        identifiers = list(splits[name])
        duplicates = sorted({item for item in identifiers if identifiers.count(item) > 1})
        if duplicates:
            raise CellLeakageError(f"сплит {name!r}: дубликаты внутри сплита: {duplicates[:10]}")
    for i, first in enumerate(names):
        for second in names[i + 1 :]:
            common = sorted(set(splits[first]) & set(splits[second]))
            if common:
                raise CellLeakageError(
                    f"утечка ячеек между {first!r} и {second!r}: {common[:10]}"
                    + (f" (всего {len(common)})" if len(common) > 10 else "")
                )
    if cell_to_batch is None:
        return
    batches: dict[str, set[str]] = {}
    for name in names:
        batches[name] = {cell_to_batch[cell] for cell in splits[name] if cell in cell_to_batch}
    for i, first in enumerate(names):
        for second in names[i + 1 :]:
            common = sorted(batches[first] & batches[second])
            if common:
                raise CellLeakageError(
                    f"утечка партий между {first!r} и {second!r}: {common[:10]}"
                    + (f" (всего {len(common)})" if len(common) > 10 else "")
                )
