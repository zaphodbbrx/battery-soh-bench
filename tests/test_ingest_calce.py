"""Smoke-тест адаптера CALCE на реальных файлах (ТЗ §14).

Фикстуры — один файл Arbin (Excel) и один CADEX (``.txt``), скопированные из
сырого набора скриптом ``scripts/make_fixtures.py``. Оба формата обязаны
разбираться одним и тем же кодом: §17.1 отдельно предупреждает про CS2_8 и
CS2_21, которые лежат в ``.txt``.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from bsb.config import BenchmarkConfig
from bsb.ingest import CalceAdapter
from bsb.ingest.base import classify_step, validate_raw_bundle
from bsb.schema import CellBundle, validate_bundle
from bsb.store import compact, read_cells, read_cycles, write_bundle

FIXTURES = Path(__file__).resolve().parent / "data"
ARBIN_FIXTURE = FIXTURES / "calce_CS2_33_arbin.xlsx"
CADEX_FIXTURE = FIXTURES / "calce_CS2_8_cadex.txt"

pytestmark = pytest.mark.skipif(
    not (ARBIN_FIXTURE.is_file() and CADEX_FIXTURE.is_file()),
    reason="нет фикстур CALCE (§14)",
)


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    """Каталог сырых данных с двумя форматами в раскладке CALCE."""
    target = tmp_path / "calce"
    for fixture in (ARBIN_FIXTURE, CADEX_FIXTURE):
        cell_dir = target / fixture.stem.split("_")[1] / fixture.stem.split("_")[1]
        cell_dir.mkdir(parents=True, exist_ok=True)
        # имя файла восстанавливаем из фикстуры: адаптер читает идентификатор
        # ячейки из имени, а не из каталога
        name = "CS2_33_10_05_10.xlsx" if fixture.suffix == ".xlsx" else "CS2_8_1_28_10.txt"
        shutil.copy2(fixture, cell_dir / name)
    return target


def test_discover_finds_both_cells(raw_dir: Path, repo_config: BenchmarkConfig) -> None:
    """Обе ячейки находятся, серии берутся из конфига."""
    sources = CalceAdapter(repo_config).discover(raw_dir)
    assert [source.cell_id for source in sources] == ["calce_CS2_33", "calce_CS2_8"]


@pytest.mark.parametrize("cell_id", ["calce_CS2_33", "calce_CS2_8"])
def test_real_files_parse_and_label(
    raw_dir: Path, repo_config: BenchmarkConfig, cell_id: str
) -> None:
    """Оба формата дают валидный бандл, размеченный по §5."""
    adapter = CalceAdapter(repo_config)
    source = next(item for item in adapter.discover(raw_dir) if item.cell_id == cell_id)
    raw = adapter.load_cell(source)
    assert validate_raw_bundle(raw) is not None

    bundle = adapter.mark_reference_cycles(raw)
    assert validate_bundle(bundle) is not None
    assert bundle.cycles.height > 0
    assert bundle.cycles["soh_source"].unique().to_list() == ["measured"]
    assert bundle.cell["form_factor"] == "prismatic"
    assert bundle.cell["chemistry"] == "LCO"
    assert bundle.cell["nominal_capacity_ah"] == pytest.approx(1.1)
    # bol — ёмкость первого полного цикла этой ячейки (§5 п.1)
    assert 0.5 < bundle.cell["bol_capacity_ah"] < 1.3


def test_charge_phase_is_cc_then_cv(raw_dir: Path, repo_config: BenchmarkConfig) -> None:
    """Заряд размечается как CC + CV, разряд — как dch (§4.4)."""
    adapter = CalceAdapter(repo_config)
    source = next(item for item in adapter.discover(raw_dir) if item.cell_id == "calce_CS2_33")
    raw = adapter.load_cell(source)
    kinds = set(raw.timeseries["step_type"].unique().to_list())
    assert {"chg_cc", "chg_cv", "dch"} <= kinds, f"размечено только {kinds}"
    protocol = str(raw.cell["charge_protocol"])
    assert "0.5" in protocol and "CV" in protocol


def test_cadex_units_are_minutes_and_milli(repo_config: BenchmarkConfig) -> None:
    """CADEX: ``Time`` в минутах, ``mV``/``mA`` — иначе ток был бы в 1000 раз больше."""
    adapter = CalceAdapter(repo_config)
    frame = adapter._read_cadex(CADEX_FIXTURE)
    assert frame["voltage_v"].max() < 5.0, "напряжение должно быть в вольтах"
    assert frame["current_a"].abs().max() < 3.0, "ток должен быть в амперах"
    assert frame["t_s"].max() > 60.0, "время должно быть в секундах"


def test_arbin_ignores_summary_sheets(repo_config: BenchmarkConfig) -> None:
    """Лист ``Statistics_*`` — не лог данных и не должен попадать в разбор."""
    adapter = CalceAdapter(repo_config)
    frame = adapter._read_arbin(ARBIN_FIXTURE)
    assert frame.height > 1000
    # Test_Time(s) идёт от старта файла и монотонен; к нулю его приводит
    # split_into_steps, отсчитывая время от начала шага (§4.3).
    assert frame["t_s"].is_sorted()
    assert frame["t_s"].min() > 0.0
    steps, dropped = adapter._steps_of(frame, max_gap_s=600.0)
    assert steps
    assert dropped >= 0, "пустые строки листа должны отбрасываться"
    for step in steps:
        assert step["t_s"][0] == pytest.approx(0.0, abs=1e-6)
        assert step["t_s"].is_sorted()


def test_reference_requires_full_cycle(raw_dir: Path, repo_config: BenchmarkConfig) -> None:
    """Опорным считается только полный цикл (заряд до верха и разряд до низа)."""
    adapter = CalceAdapter(repo_config)
    source = next(item for item in adapter.discover(raw_dir) if item.cell_id == "calce_CS2_33")
    raw = adapter.load_cell(source)
    for cycle in raw.cycles.filter(raw.cycles["is_reference"]).head(3).iter_rows(named=True):
        part = raw.timeseries.filter(part_cycle(cycle["cycle_index"]))
        charge = part.filter(part["step_type"].is_in(["chg_cc", "chg_cv"]))
        discharge = part.filter(part["step_type"] == "dch")
        assert charge["voltage_v"].max() >= 4.15
        assert discharge["voltage_v"].min() <= 2.75


def part_cycle(cycle_index: int):  # type: ignore[no-untyped-def]
    """Условие отбора одного цикла (вынесено ради читаемости теста)."""
    import polars as pl

    return pl.col("cycle_index") == cycle_index


def test_cycles_round_trip_through_store(
    raw_dir: Path,
    make_config: Callable[[dict[str, Any] | None], BenchmarkConfig],
) -> None:
    """Ячейка пишется в хранилище и читается обратно."""
    config = make_config()
    adapter = CalceAdapter(config)
    source = next(item for item in adapter.discover(raw_dir) if item.cell_id == "calce_CS2_33")
    bundle = adapter.mark_reference_cycles(adapter.load_cell(source))
    write_bundle(config, bundle, dataset="calce")
    compact(config, datasets=["calce"])
    assert read_cells(config).height == 1
    assert read_cycles(config).height == bundle.cycles.height


# --------------------------------------------------------------------------
# §4.4: общий классификатор шагов
# --------------------------------------------------------------------------


def test_classify_step_rules(repo_config: BenchmarkConfig) -> None:
    """Классификатор различает CC, CV, разряд и отдых по данным, а не по номеру."""
    import numpy as np

    labeling = repo_config.step_labeling
    kwargs = {
        "cc_current_tolerance_rel": labeling.cc_current_tolerance_rel,
        "cv_voltage_tolerance_v": labeling.cv_voltage_tolerance_v,
        "rest_current_tolerance_a": labeling.rest_current_tolerance_a,
        "current_sign_tolerance_a": labeling.current_sign_tolerance_a,
    }
    charge = np.full(50, 0.55)
    assert classify_step(charge, np.linspace(3.4, 4.2, 50), **kwargs) == "chg_cc"
    taper = np.linspace(0.55, 0.05, 20)
    assert classify_step(taper, np.full(20, 4.2), **kwargs) == "chg_cv"
    assert classify_step(np.full(20, -0.55), np.linspace(4.1, 2.7, 20), **kwargs) == "dch"
    assert classify_step(np.zeros(10), np.full(10, 3.9), **kwargs) == "rest"
    tiny = np.full(10, 0.001)
    assert classify_step(tiny, np.full(10, 3.9), **kwargs) == "rest"


def test_cell_bundle_type_exported() -> None:
    """Адаптер возвращает именно тот тип, который ждёт хранилище (§6)."""
    assert CellBundle.__name__ == "CellBundle"
