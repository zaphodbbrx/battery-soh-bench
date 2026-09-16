"""Участнический доступ к данным: бандлы ячеек и фрагменты (ТЗ §7).

Модуль отвечает за две вещи:

* собрать :class:`~bsb.schema.CellBundle` из унифицированного хранилища;
* отдать фрагменты — готовые (``fixed``) или нарисованные на лету
  (``sampling``) для аугментации на обучении.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import numpy as np
import polars as pl

from bsb.config import BenchmarkConfig
from bsb.fragments import fragment_id, iter_sampled_windows
from bsb.schema import (
    FRAGMENTS_SCHEMA,
    FRAGMENTS_UNLABELED_SCHEMA,
    CellBundle,
    conform,
)
from bsb.store import read_cells, read_cycles, read_timeseries

__all__ = [
    "load_bundle",
    "load_split_fragments",
    "read_split_cell_ids",
    "sample_fragments",
]


def load_bundle(config: BenchmarkConfig, cell_id: str) -> CellBundle:
    """Собрать бандл одной ячейки из унифицированного уровня."""
    cells = read_cells(config).filter(pl.col("cell_id") == cell_id)
    if cells.height == 0:
        raise FileNotFoundError(
            f"{cell_id}: нет в унифицированном уровне {config.paths.unified_dir}"
        )
    cycles = read_cycles(config, cell_ids=[cell_id])
    timeseries = read_timeseries(config, cell_id)
    return CellBundle(cells=cells, cycles=cycles, timeseries=timeseries)


def iter_bundles(config: BenchmarkConfig, cell_ids: Iterable[str]) -> Iterator[CellBundle]:
    """По одной ячейке за раз: KIT целиком в память не помещается (ТЗ §6.3)."""
    for cell_id in cell_ids:
        yield load_bundle(config, cell_id)


def read_split_cell_ids(config: BenchmarkConfig, split: str) -> list[str]:
    """Прочитать список ячеек сплита из ``splits/*.txt`` (ТЗ §8)."""
    from bsb.splits import read_split, split_path

    return read_split(split_path(config, split))


def load_split_fragments(
    config: BenchmarkConfig,
    split: str,
    *,
    labeled: bool | None = None,
) -> pl.DataFrame:
    """Загрузить готовые (``fixed``) фрагменты сплита (ТЗ §7).

    Parameters
    ----------
    labeled
        ``None`` — по умолчанию: обучающие сплиты с метками, тест без меток.
    """
    path = config.paths.fragments_path(split, labeled=labeled)
    if not path.is_file():
        raise FileNotFoundError(
            f"{path}: фрагменты не сгенерированы (bsb fragments --split {split})"
        )
    schema = (
        FRAGMENTS_SCHEMA if path.name != "test_unlabeled.parquet" else FRAGMENTS_UNLABELED_SCHEMA
    )
    return conform(pl.read_parquet(path), schema, name=f"fragments[{split}]")


def sample_fragments(
    bundles: Iterable[CellBundle],
    config: BenchmarkConfig,
    *,
    windows: int = 1,
    rng: np.random.Generator | None = None,
    labeled: bool = True,
) -> pl.DataFrame:
    """Нарисовать случайные окна для аугментации на обучении (ТЗ §7, ``sampling``).

    В отличие от ``fixed`` результат намеренно не воспроизводим: окно берётся
    из ``rng``, а не из ``fragment.seed``. Именно поэтому такие фрагменты нельзя
    коммитить как артефакт val/test.
    """
    generator = rng if rng is not None else np.random.default_rng()
    rows: list[dict[str, object]] = []
    for bundle in bundles:
        sampled = iter_sampled_windows(bundle, config, generator, windows=windows)
        for window in sampled:
            soh = float(bundle.cycles.filter(pl.col("cycle_index") == window.cycle_index)["soh"][0])
            seed = int(generator.integers(0, 2**31 - 1))
            rows.append(
                {
                    "fragment_id": fragment_id(
                        bundle.cell_id, window.cycle_index, window.v_start, seed
                    ),
                    "cell_id": bundle.cell_id,
                    "cycle_index": window.cycle_index,
                    "v_start": window.v_start,
                    "v_end": window.v_end,
                    "q_norm": window.q_norm.tolist(),
                    "current": window.current.tolist(),
                    "temperature": window.temperature.tolist(),
                    "soh": soh,
                }
            )
    if not rows:
        return pl.DataFrame(
            schema=dict(FRAGMENTS_SCHEMA if labeled else FRAGMENTS_UNLABELED_SCHEMA)
        )
    frame = pl.DataFrame(rows, schema=dict(FRAGMENTS_SCHEMA))
    if not labeled:
        frame = frame.drop(["cell_id", "cycle_index", "soh"])
    return conform(
        frame,
        FRAGMENTS_SCHEMA if labeled else FRAGMENTS_UNLABELED_SCHEMA,
        name="fragments (sampling)",
    )
