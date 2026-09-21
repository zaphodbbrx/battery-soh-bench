"""Smoke-тест адаптера архива Sandia R&D (Barkholtz et al.) — ТЗ §14.

Фикстура ``tests/data/sandia_rd_NCA_Reg_sample.xls`` — самый маленький файл
архива из суффикса ``_Reg``, дающий больше одного опорного цикла. Файлы ``_Mod``
для этого не годятся: это импульсные тесты скоростных возможностей, и адаптер
отвергает их по балансу заряда и разряда (см. ``mark_reference_cycles``).
Источник — R&D Data Repository Sandia, ``scripts/download_sandia_rd.sh``.

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
from bsb.ingest.base import charge_throughput_by_cycle, validate_raw_bundle
from bsb.labels import NoReferenceCycles
from bsb.schema import validate_bundle
from bsb.store import compact, read_cells, write_bundle

FIXTURE = Path(__file__).resolve().parent / "data" / "sandia_rd_NCA_Reg_sample.xls"

pytestmark = pytest.mark.skipif(not FIXTURE.is_file(), reason="нет фикстуры Sandia R&D (§14)")


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    """Каталог с одним файлом архива — как после scripts/download_sandia_rd.sh."""
    target = tmp_path / "sandia_rd" / "NCA_6"
    target.mkdir(parents=True)
    shutil.copy2(FIXTURE, target / "NCA_6_35C_Reg.xls")
    return tmp_path / "sandia_rd"


def test_discover_and_metadata(raw_dir: Path, repo_config: BenchmarkConfig) -> None:
    """Химия, температура и физическая ячейка берутся из имени файла."""
    sources = SandiaRdAdapter(repo_config).discover(raw_dir)
    assert [source.cell_id for source in sources] == ["sandia_rd_NCA_6_35C_Reg"]
    assert sources[0].extras["physical_cell"] == "NCA_6"


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
    assert bundle.cell["chemistry"] == "NCA"
    assert bundle.cell["temperature_setpoint_c"] == pytest.approx(35.0)
    # batch_id — физическая ячейка: она тестировалась при нескольких
    # температурах, и §8 разрешает разбивать по партии, чтобы её записи не
    # разъехались по train и val
    assert bundle.cell["batch_id"] == "NCA_6"
    assert bundle.cycles.height >= 2
    assert bundle.cycles["soh_source"].unique().to_list() == ["measured"]


def test_reference_cycle_is_full_and_modal(raw_dir: Path, repo_config: BenchmarkConfig) -> None:
    """Опорный цикл: полный размах, модальный ток и сбалансированный заряд."""
    adapter = SandiaRdAdapter(repo_config)
    raw = adapter.load_cell(adapter.discover(raw_dir)[0])
    stats = adapter._cycle_stats(raw.timeseries)
    throughput = charge_throughput_by_cycle(raw.timeseries)
    top = float(stats["v_charge_max"].max())
    bottom = float(stats["v_discharge_min"].min())
    modal = adapter._modal_rate(stats["i_discharge"])
    bundle = adapter.mark_reference_cycles(raw)
    references = bundle.cycles.filter(pl.col("is_reference"))
    assert references.height >= 2
    for row in references.iter_rows(named=True):
        record = stats.filter(pl.col("cycle_index") == row["cycle_index"]).row(0, named=True)
        assert record["v_charge_max"] >= top - 0.05
        assert record["v_discharge_min"] <= bottom + 0.05
        assert record["i_discharge"] == pytest.approx(modal, rel=0.02)
        # Баланс: разряд вернул примерно столько же, сколько вошло при заряде.
        charge = throughput.filter(pl.col("cycle_index") == row["cycle_index"])["q_charge"][0]
        assert abs(row["capacity_ah"] / charge - 1.0) <= 0.35
        # И скорость разряда физически возможна.
        assert record["i_discharge"] / row["capacity_ah"] <= 3.0


def test_pulse_discharge_is_not_a_reference_cycle(repo_config: BenchmarkConfig) -> None:
    """Импульсный разряд не должен становиться опорным циклом.

    Так устроены 78 файлов архива с суффиксом ``_Mod``: заряд переносит
    0.865 А·ч, а «разряд» — импульс 20 А на 40 с (0.222 А·ч). Пока баланс не
    проверялся, такие циклы попадали в опорные, и SOH выходил ровно 1.0
    (а иногда 1.17, если импульс оказывался длиннее первого).
    """
    from bsb.ingest.base import build_raw_bundle

    adapter = SandiaRdAdapter(repo_config)
    rows = []
    for index in (1, 2):
        # Заряд: 0.865 А·ч за 2841 с при 1.1 А — полный размах 3.26-3.60 В.
        for step in range(286):
            rows.append(
                (
                    index,
                    "chg_cc",
                    float(step * 10),
                    3.26 + 0.34 * step / 285,
                    1.0997,
                    0.865 * step / 285,
                )
            )
        # «Разряд»: импульс 20 А на 40 с — те же 0.222 А·ч, что и в архиве.
        for step in range(41):
            rows.append(
                (
                    index,
                    "dch",
                    float(step * 1),
                    3.05 - 0.735 * step / 40,
                    -20.003,
                    -0.222 * step / 40,
                )
            )
    timeseries = pl.DataFrame(
        rows,
        orient="row",
        schema={
            "cycle_index": pl.Int32,
            "step_type": pl.String,
            "t_s": pl.Float64,
            "voltage_v": pl.Float64,
            "current_a": pl.Float64,
            "charge_ah": pl.Float64,
        },
    ).with_columns(
        pl.lit("sandia_rd_TEST").alias("cell_id"),
        pl.lit(25.0).alias("temperature_c"),
        pl.col("step_type").cast(pl.Enum(["chg_cc", "chg_cv", "dch", "rest"])),
    )
    cycles = pl.DataFrame(
        {
            "cell_id": ["sandia_rd_TEST"] * 2,
            "cycle_index": [1, 2],
            "capacity_ah": [0.222, 0.222],
            "is_reference": [True, True],
        }
    )
    cell = {
        "cell_id": "sandia_rd_TEST",
        "dataset": "sandia_rd",
        "chemistry": "LFP",
        "form_factor": "cylindrical_18650",
        "nominal_capacity_ah": 0.222,
        "temperature_setpoint_c": 25.0,
        "charge_protocol": "CC-CV",
        "batch_id": "TEST",
        "notes": None,
    }
    bundle = build_raw_bundle(cell, cycles, timeseries=timeseries)
    with pytest.raises(NoReferenceCycles):
        adapter.mark_reference_cycles(bundle)


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
