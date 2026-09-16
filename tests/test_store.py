"""Тесты хранилища унифицированного уровня (ТЗ §3, §4)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import polars as pl
import pytest

from bsb.config import BenchmarkConfig
from bsb.schema import CellBundle
from bsb.store import (
    compact,
    list_cells,
    parts_dir,
    read_cells,
    read_cycles,
    read_timeseries,
    write_bundle,
)
from conftest import synth_bundle


def test_write_then_compact_roundtrip(
    make_config: Callable[[dict[str, Any] | None], BenchmarkConfig],
    synthetic_bundle: CellBundle,
) -> None:
    """ingest -> parts -> compact -> чтение возвращает те же данные."""
    config = make_config()
    write_bundle(config, synthetic_bundle, dataset="oxford")
    assert (parts_dir(config, "oxford") / "cells.parquet").is_file()

    compact(config, datasets=["oxford"])

    cells = read_cells(config)
    assert cells.height == 1
    assert cells["cell_id"].to_list() == [synthetic_bundle.cell_id]
    cycles = read_cycles(config)
    assert cycles.height == synthetic_bundle.cycles.height
    timeseries = read_timeseries(config, synthetic_bundle.cell_id)
    assert timeseries.height == synthetic_bundle.timeseries.height
    assert timeseries.columns == synthetic_bundle.timeseries.columns
    assert list_cells(config) == [synthetic_bundle.cell_id]


def test_reingest_is_idempotent(
    make_config: Callable[[dict[str, Any] | None], BenchmarkConfig],
) -> None:
    """Повторный ingest ячейки заменяет её строки, а не дублирует."""
    config = make_config()
    write_bundle(config, synth_bundle(n_cycles=4), dataset="oxford")
    write_bundle(config, synth_bundle(n_cycles=9), dataset="oxford")
    compact(config, datasets=["oxford"])
    assert read_cells(config).height == 1
    assert read_cycles(config).height == 9


def test_compact_merges_datasets(
    make_config: Callable[[dict[str, Any] | None], BenchmarkConfig],
) -> None:
    """Один и тот же код обслуживает train и test (ТЗ §6.4)."""
    config = make_config()
    write_bundle(config, synth_bundle(), dataset="oxford")
    write_bundle(
        config,
        synth_bundle(
            cell_id="sandia_Cell1",
            dataset="sandia",
            chemistry="NCA",
            form_factor="cylindrical_18650",
            nominal_capacity_ah=3.2,
            bol_capacity_ah=3.05,
            temperature_setpoint_c=25.0,
        ),
        dataset="sandia",
    )
    compact(config)
    assert read_cells(config).height == 2
    assert read_cells(config, datasets=["sandia"])["cell_id"].to_list() == ["sandia_Cell1"]
    assert read_cycles(config, cell_ids=["oxford_Cell1"]).height == 6


def test_compact_reports_unknown_dataset(
    make_config: Callable[[dict[str, Any] | None], BenchmarkConfig],
) -> None:
    """Запрошенный, но не разобранный датасет — внятная ошибка."""
    config = make_config()
    write_bundle(config, synth_bundle(), dataset="oxford")
    with pytest.raises(FileNotFoundError, match="kit"):
        compact(config, datasets=["kit"])


def test_compact_without_parts_is_reported(
    make_config: Callable[[dict[str, Any] | None], BenchmarkConfig],
) -> None:
    """Без ingest команда compact говорит, что делать."""
    config = make_config()
    with pytest.raises(FileNotFoundError, match="bsb ingest"):
        compact(config)


def test_read_missing_table_is_reported(
    make_config: Callable[[dict[str, Any] | None], BenchmarkConfig],
) -> None:
    """Чтение до compact объясняет причину."""
    config = make_config()
    with pytest.raises(FileNotFoundError, match="compact"):
        read_cells(config)


def test_write_bundle_rejects_multirow_cells(
    make_config: Callable[[dict[str, Any] | None], BenchmarkConfig],
    synthetic_bundle: CellBundle,
) -> None:
    """Бандл с двумя строками cells не пишется."""
    config = make_config()
    doubled = pl.concat([synthetic_bundle.cells, synthetic_bundle.cells], how="vertical")
    broken = CellBundle(
        cells=doubled, cycles=synthetic_bundle.cycles, timeseries=synthetic_bundle.timeseries
    )
    with pytest.raises(ValueError, match="одна строка"):
        write_bundle(config, broken, dataset="oxford")
