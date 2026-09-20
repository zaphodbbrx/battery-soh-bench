"""Smoke-тест адаптера архива Sandia R&D (Barkholtz et al.) — ТЗ §14.

Фикстура ``tests/data/sandia_rd_NMC_sample.xls`` — самый маленький файл архива,
который даёт больше одного опорного цикла (файлы 45 °C содержат только
частичные циклы, и адаптер их обоснованно пропускает). Источник — R&D Data
Repository Sandia, ``scripts/download_sandia_rd.sh``.

Датасет идёт **только в обучение**: это другое исследование, чем SNL/Preger
(скрытый тест), поэтому и назван отдельно (``sandia_rd``).
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from bsb.config import BenchmarkConfig
from bsb.ingest import SandiaRdAdapter
from bsb.ingest.base import validate_raw_bundle
from bsb.labels import NoReferenceCycles
from bsb.schema import validate_bundle
from bsb.store import compact, read_cells, write_bundle

FIXTURE = Path(__file__).resolve().parent / "data" / "sandia_rd_NMC_sample.xls"

pytestmark = pytest.mark.skipif(not FIXTURE.is_file(), reason="нет фикстуры Sandia R&D (§14)")


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    """Каталог с одним файлом архива — как после scripts/download_sandia_rd.sh."""
    target = tmp_path / "sandia_rd" / "NMC_1_5C-15C"
    target.mkdir(parents=True)
    shutil.copy2(FIXTURE, target / "NMC_1_15C_Mod.xls")
    return tmp_path / "sandia_rd"


def test_discover_and_metadata(raw_dir: Path, repo_config: BenchmarkConfig) -> None:
    """Химия, температура и физическая ячейка берутся из имени файла."""
    sources = SandiaRdAdapter(repo_config).discover(raw_dir)
    assert [source.cell_id for source in sources] == ["sandia_rd_NMC_1_15C_Mod"]
    assert sources[0].extras["physical_cell"] == "NMC_1"


def test_real_file_parses_and_labels(raw_dir: Path, repo_config: BenchmarkConfig) -> None:
    """Реальный файл проходит ingest -> labels и валидаторы §4."""
    adapter = SandiaRdAdapter(repo_config)
    raw = adapter.load_cell(adapter.discover(raw_dir)[0])
    assert validate_raw_bundle(raw) == []

    bundle = adapter.mark_reference_cycles(raw)
    warnings = validate_bundle(bundle)
    assert all("вне бенчмарка" in warning or "soh > 1.0" in warning for warning in warnings), (
        warnings
    )
    assert bundle.cell["dataset"] == "sandia_rd"
    assert bundle.cell["chemistry"] == "NMC"
    assert bundle.cell["temperature_setpoint_c"] == pytest.approx(15.0)
    # batch_id — физическая ячейка: она тестировалась при нескольких
    # температурах, и §8 разрешает разбивать по партии, чтобы её записи не
    # разъехались по train и val
    assert bundle.cell["batch_id"] == "NMC_1"
    assert bundle.cycles.height >= 2
    assert bundle.cycles["soh_source"].unique().to_list() == ["measured"]


def test_reference_cycle_is_full_and_modal(raw_dir: Path, repo_config: BenchmarkConfig) -> None:
    """Опорный цикл: полный размах по напряжению и модальный ток разряда."""
    adapter = SandiaRdAdapter(repo_config)
    raw = adapter.load_cell(adapter.discover(raw_dir)[0])
    stats = adapter._cycle_stats(raw.timeseries)
    top = float(stats["v_charge_max"].max())
    bottom = float(stats["v_discharge_min"].min())
    modal = adapter._modal_rate(stats["i_discharge"])
    for row in raw.cycles.filter(raw.cycles["is_reference"]).iter_rows(named=True):
        record = stats.filter(pl.col("cycle_index") == row["cycle_index"]).row(0, named=True)
        assert record["v_charge_max"] >= top - 0.05
        assert record["v_discharge_min"] <= bottom + 0.05
        assert record["i_discharge"] == pytest.approx(modal, rel=0.02)


def test_partial_only_file_is_skipped_not_failed(repo_config: BenchmarkConfig) -> None:
    """Файл только с частичными циклами — это пропуск, а не ошибка разбора.

    В архиве так устроены файлы 45 °C (мелкие SOC-качели по 0.2-0.35 В).
    CLI отличает такой случай от настоящей ошибки.
    """
    import polars as pl

    from bsb.ingest.base import build_raw_bundle

    adapter = SandiaRdAdapter(repo_config)
    partial = pl.DataFrame(
        {
            "cell_id": ["sandia_rd_TEST"] * 4,
            "cycle_index": [1, 1, 2, 2],
            "step_type": ["chg_cc", "dch", "chg_cc", "dch"],
            "t_s": [0.0, 10.0, 0.0, 10.0],
            "voltage_v": [3.6, 3.4, 3.6, 3.4],
            "current_a": [1.0, -1.0, 1.0, -1.0],
            "temperature_c": [45.0] * 4,
            "charge_ah": [0.0, -0.01, 0.0, -0.01],
        }
    )
    cycles = pl.DataFrame(
        {
            "cell_id": ["sandia_rd_TEST"] * 2,
            "cycle_index": [1, 2],
            "capacity_ah": [0.01, 0.01],
            "is_reference": [True, True],
        }
    )
    cell = {
        "cell_id": "sandia_rd_TEST",
        "dataset": "sandia_rd",
        "chemistry": "NMC",
        "form_factor": "cylindrical_18650",
        "nominal_capacity_ah": 2.5,
        "temperature_setpoint_c": 45.0,
        "charge_protocol": "CC-CV",
        "batch_id": "TEST",
        "notes": None,
    }
    bundle = build_raw_bundle(cell, cycles, timeseries=partial)
    with pytest.raises(NoReferenceCycles):
        adapter.mark_reference_cycles(bundle)


def test_round_trip_through_store(
    raw_dir: Path,
    make_config: Callable[[dict[str, Any] | None], BenchmarkConfig],
) -> None:
    """Запись архива пишется в хранилище и читается обратно под своим датасетом."""
    config = make_config()
    adapter = SandiaRdAdapter(config)
    bundle = adapter.mark_reference_cycles(adapter.load_cell(adapter.discover(raw_dir)[0]))
    write_bundle(config, bundle, dataset="sandia_rd")
    compact(config, datasets=["sandia_rd"])
    cells = read_cells(config)
    assert cells.height == 1
    assert cells["dataset"].to_list() == ["sandia_rd"]
