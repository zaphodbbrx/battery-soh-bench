"""Smoke-тест адаптера Oxford на реальной ячейке (ТЗ §14).

Фикстура ``tests/data/oxford_Cell1_fixture.mat`` вырезана из сырого набора
скриптом ``scripts/make_fixtures.py`` (cyc0000 = начало жизни, cyc3200 =
аномалия со склейкой сегментов, cyc8200 = конец жизни) и зафиксирована в
репозитории, поэтому тест идёт в CI без скачивания 253 МБ.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from bsb.config import BenchmarkConfig
from bsb.ingest import OxfordAdapter
from bsb.schema import validate_bundle
from bsb.store import compact, read_cells, read_cycles, read_timeseries, write_bundle

FIXTURE = Path(__file__).resolve().parent / "data" / "oxford_Cell1_fixture.mat"

pytestmark = pytest.mark.skipif(not FIXTURE.is_file(), reason="нет фикстуры Oxford (§14)")


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    """Каталог с единственным ``.mat`` — как у настоящего датасета."""
    target = tmp_path / "raw"
    target.mkdir()
    shutil.copy2(FIXTURE, target / FIXTURE.name)
    return target


def test_discover_finds_cell(raw_dir: Path, repo_config: BenchmarkConfig) -> None:
    """Адаптер находит ячейку и не путается в именах файлов."""
    sources = OxfordAdapter(repo_config).discover(raw_dir)
    assert [source.cell_id for source in sources] == ["oxford_Cell1"]


def test_real_cell_passes_validation_and_labels(
    raw_dir: Path, repo_config: BenchmarkConfig
) -> None:
    """Реальная ячейка проходит ingest -> labels и валидаторы §4 без замечаний."""
    adapter = OxfordAdapter(repo_config)
    bundle = adapter.mark_reference_cycles(adapter.load_cell(adapter.discover(raw_dir)[0]))

    assert validate_bundle(bundle) == []
    assert bundle.cycles.height == 3
    assert bundle.cycles["soh_source"].unique().to_list() == ["measured"]
    assert bundle.cycles["soh_gap_cycles"].unique().to_list() == [0]

    # bol — ёмкость первого характеризационного разряда (C/18.5, ~740 мА·ч).
    assert bundle.cell["bol_capacity_ah"] == pytest.approx(0.7398, abs=1e-3)
    assert bundle.cycles["soh"].to_list() == sorted(bundle.cycles["soh"].to_list(), reverse=True)
    assert bundle.cycles["soh"].min() == pytest.approx(0.7689, abs=1e-3)

    # Порядок циклов: cyc0000 -> cyc3200 -> cyc8200 (сдвиг на 1 по §4.2).
    assert bundle.cycles["cycle_index"].to_list() == [1, 3201, 8201]


def test_adapter_reports_recording_seam(raw_dir: Path, repo_config: BenchmarkConfig) -> None:
    """Аномалия склейки сегментов попадает в notes, а не замалчивается (§4.1)."""
    adapter = OxfordAdapter(repo_config)
    bundle = adapter.mark_reference_cycles(adapter.load_cell(adapter.discover(raw_dir)[0]))
    notes = bundle.cell["notes"]
    assert notes is not None
    assert "cyc3200.OCVch" in notes
    assert "оставлен последний" in notes


def test_real_cell_round_trips_through_store(
    raw_dir: Path,
    make_config: Callable[[dict[str, Any] | None], BenchmarkConfig],
) -> None:
    """Реальная ячейка пишется в хранилище и читается обратно без потерь."""
    config = make_config()
    adapter = OxfordAdapter(config)
    bundle = adapter.mark_reference_cycles(adapter.load_cell(adapter.discover(raw_dir)[0]))
    write_bundle(config, bundle, dataset="oxford")
    compact(config, datasets=["oxford"])

    assert read_cells(config).height == 1
    assert read_cycles(config).height == bundle.cycles.height
    timeseries = read_timeseries(config, bundle.cell_id)
    assert timeseries.height == bundle.timeseries.height
    charge_step = config.ingest.oxford.charge_step
    charge = timeseries.filter(
        (timeseries["cycle_index"] == 8201) & (timeseries["step_type"] == "chg_cc")
    )
    assert charge.height > 10_000, f"{charge_step} должен попадать в timeseries целиком"


def test_switching_charge_step_changes_ingested_timeseries(
    raw_dir: Path,
    make_config: Callable[[dict[str, Any] | None], BenchmarkConfig],
) -> None:
    """Выбор подшага заряда — параметр конфига, а не захардкоженная константа."""
    oxford_config = make_config({"ingest.oxford.charge_step": "C1ch"})
    adapter = OxfordAdapter(oxford_config)
    bundle = adapter.mark_reference_cycles(adapter.load_cell(adapter.discover(raw_dir)[0]))
    median_current = float(
        bundle.timeseries.filter(bundle.timeseries["step_type"] == "chg_cc")["current_a"].median()
    )
    assert median_current == pytest.approx(0.740, abs=0.01), "должен использоваться заряд 1C"
