"""Хранилище унифицированного уровня (ТЗ §3, §4).

Раскладка на диске::

    data/unified/
      parts/<dataset>/cells.parquet          # промежуточный слой, наполняется ingest
      parts/<dataset>/cycles.parquet
      parts/<dataset>/timeseries/<cell_id>.parquet
      cells.parquet                          # результат compact()
      cycles.parquet
      timeseries/<cell_id>.parquet

Адаптеры пишут только в ``parts/`` (по одному датасету), ``compact()``
собирает общий уровень. Перезапись идемпотентна по ``cell_id``: повторный
ingest ячейки заменяет её строки, а не дублирует их.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import polars as pl

from bsb.config import BenchmarkConfig
from bsb.schema import (
    CELLS_SCHEMA,
    CYCLES_SCHEMA,
    TIMESERIES_SCHEMA,
    CellBundle,
    conform,
)

__all__ = [
    "compact",
    "list_cells",
    "parts_dir",
    "read_cells",
    "read_cycles",
    "read_timeseries",
    "scan_timeseries",
    "write_bundle",
]


def parts_dir(config: BenchmarkConfig, dataset: str | None = None) -> Path:
    """Каталог промежуточного слоя (всего или одного датасета)."""
    root = config.paths.unified_dir / "parts"
    return root / dataset if dataset else root


def _upsert(path: Path, frame: pl.DataFrame, *, key: str) -> pl.DataFrame:
    """Заменить в файле строки с ключами из ``frame`` и вернуть новый кадр."""
    if path.is_file():
        existing = pl.read_parquet(path)
        stale = set(frame[key].to_list())
        existing = existing.filter(~pl.col(key).is_in(list(stale)))
        if existing.height:
            frame = pl.concat([existing, frame], how="vertical")
    return frame.sort(key)


def write_bundle(
    config: BenchmarkConfig, bundle: CellBundle, *, dataset: str | None = None
) -> Path:
    """Записать бандл ячейки в промежуточный слой ``parts/<dataset>``.

    Returns
    -------
    Path
        Каталог датасета, в который записан бандл.
    """
    name = dataset or str(bundle.cell["dataset"])
    target = parts_dir(config, name)
    (target / "timeseries").mkdir(parents=True, exist_ok=True)
    cells = conform(bundle.cells, CELLS_SCHEMA, name="cells")
    cycles = conform(bundle.cycles, CYCLES_SCHEMA, name="cycles")
    timeseries = conform(bundle.timeseries, TIMESERIES_SCHEMA, name="timeseries")
    if cells.height != 1:
        raise ValueError(f"write_bundle: ожидалась одна строка cells, получено {cells.height}")
    cell_id = str(cells["cell_id"][0])
    if cell_id != bundle.cell_id:
        raise ValueError(
            f"write_bundle: cell_id кадров ({cell_id}) и бандла ({bundle.cell_id}) разошлись"
        )
    _upsert(target / "cells.parquet", cells, key="cell_id").write_parquet(target / "cells.parquet")
    _upsert(target / "cycles.parquet", cycles, key="cell_id").write_parquet(
        target / "cycles.parquet"
    )
    timeseries.write_parquet(target / "timeseries" / f"{cell_id}.parquet")
    return target


def compact(config: BenchmarkConfig, *, datasets: Sequence[str] | None = None) -> Path:
    """Собрать общий унифицированный уровень из ``parts/``.

    Один и тот же код обслуживает обучающие датасеты и тест (ТЗ §6.4):
    никаких ветвлений «если тест» здесь нет.
    """
    root = parts_dir(config)
    if not root.is_dir():
        raise FileNotFoundError(f"нет промежуточного слоя: {root}; сначала выполните bsb ingest")
    available = sorted(path.name for path in root.iterdir() if path.is_dir())
    selected = list(datasets) if datasets else available
    unknown = sorted(set(selected) - set(available))
    if unknown:
        raise FileNotFoundError(f"нет parts для датасетов {unknown}; есть {available}")
    if not selected:
        raise FileNotFoundError(f"{root} пуст: нет ни одного датасета")

    cells_parts: list[pl.DataFrame] = []
    cycles_parts: list[pl.DataFrame] = []
    timeseries_files: list[Path] = []
    for name in selected:
        cell_part = root / name / "cells.parquet"
        cycle_part = root / name / "cycles.parquet"
        if cell_part.is_file():
            cells_parts.append(pl.read_parquet(cell_part))
        if cycle_part.is_file():
            cycles_parts.append(pl.read_parquet(cycle_part))
        timeseries_files.extend(sorted((root / name / "timeseries").glob("*.parquet")))

    config.paths.unified_dir.mkdir(parents=True, exist_ok=True)
    config.paths.timeseries_dir.mkdir(parents=True, exist_ok=True)
    if cells_parts:
        cells = conform(pl.concat(cells_parts, how="vertical"), CELLS_SCHEMA, name="cells").sort(
            "cell_id"
        )
        cells.write_parquet(config.paths.unified_cells)
    if cycles_parts:
        cycles = conform(
            pl.concat(cycles_parts, how="vertical"), CYCLES_SCHEMA, name="cycles"
        ).sort(["cell_id", "cycle_index"])
        cycles.write_parquet(config.paths.unified_cycles)
    for source in timeseries_files:
        frame = conform(pl.read_parquet(source), TIMESERIES_SCHEMA, name="timeseries")
        cell_id = source.stem
        foreign = set(frame["cell_id"].unique().to_list()) - {cell_id}
        if foreign:
            raise ValueError(f"{source}: чужие cell_id {sorted(foreign)} в партиции {cell_id}")
        frame.write_parquet(config.paths.timeseries_path(cell_id))
    return config.paths.unified_dir


def _read_table(path: Path, schema: dict[str, pl.DataType], *, name: str) -> pl.DataFrame:
    """Прочитать parquet с проверкой схемы."""
    if not path.is_file():
        raise FileNotFoundError(
            f"{name} не найден: {path}; сначала выполните bsb ingest и bsb compact"
        )
    return conform(pl.read_parquet(path), schema, name=name)


def read_cells(config: BenchmarkConfig, *, datasets: Iterable[str] | None = None) -> pl.DataFrame:
    """Прочитать ``cells.parquet``, опционально отфильтровав по датасетам."""
    frame = _read_table(config.paths.unified_cells, dict(CELLS_SCHEMA), name="cells")
    if datasets is not None:
        frame = frame.filter(pl.col("dataset").is_in(list(datasets)))
    return frame


def read_cycles(
    config: BenchmarkConfig,
    *,
    cell_ids: Iterable[str] | None = None,
) -> pl.DataFrame:
    """Прочитать ``cycles.parquet``, опционально отфильтровав по ячейкам."""
    frame = _read_table(config.paths.unified_cycles, dict(CYCLES_SCHEMA), name="cycles")
    if cell_ids is not None:
        frame = frame.filter(pl.col("cell_id").is_in(list(cell_ids)))
    return frame


def read_timeseries(config: BenchmarkConfig, cell_id: str) -> pl.DataFrame:
    """Прочитать партицию ``timeseries`` одной ячейки."""
    return _read_table(
        config.paths.timeseries_path(cell_id), dict(TIMESERIES_SCHEMA), name="timeseries"
    )


def scan_timeseries(
    config: BenchmarkConfig, *, cell_ids: Iterable[str] | None = None
) -> pl.LazyFrame:
    """Лениво прочитать весь уровень ``timeseries`` (для KIT, ТЗ §6.3)."""
    pattern = str(config.paths.timeseries_dir / "*.parquet")
    frame = pl.scan_parquet(pattern)
    if cell_ids is not None:
        frame = frame.filter(pl.col("cell_id").is_in(list(cell_ids)))
    return frame


def list_cells(config: BenchmarkConfig) -> list[str]:
    """Список ``cell_id`` на унифицированном уровне (из партиций timeseries)."""
    if not config.paths.timeseries_dir.is_dir():
        return []
    return sorted(path.stem for path in config.paths.timeseries_dir.glob("*.parquet"))
