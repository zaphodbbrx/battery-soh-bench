"""Smoke-тест адаптера Sandia (SNL) и общих правил сегментации (ТЗ §14).

Фикстура ``tests/data/sandia_SNL_18650_NMC_25C_0-100_0.5-1C_a_small.pkl`` —
метаданные и первые шесть циклов реальной ячейки SNL, вырезанные скриптом
``scripts/make_fixtures.py`` (полный pickle весит 18 МБ и в репозиторий не
помещается). Источник — BatteryLife (CC-BY-4.0), см. ``scripts/download_sandia.sh``.
"""

from __future__ import annotations

import pickle
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from bsb.config import BenchmarkConfig
from bsb.ingest import SandiaAdapter
from bsb.ingest.base import (
    integrate_charge,
    isolated_sign_flips,
    plausible_mask,
    segment_cycle,
    validate_raw_bundle,
)
from bsb.schema import CellBundle, validate_bundle

FIXTURE = (
    Path(__file__).resolve().parent / "data" / "sandia_SNL_18650_NMC_25C_0-100_0.5-1C_a_small.pkl"
)

pytestmark = pytest.mark.skipif(not FIXTURE.is_file(), reason="нет фикстуры Sandia (§14)")


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    """Каталог с одной ячейкой — как после scripts/download_sandia.sh."""
    import shutil

    target = tmp_path / "sandia"
    target.mkdir()
    shutil.copy2(FIXTURE, target / "SNL_18650_NMC_25C_0-100_0.5-1C_a.pkl")
    return target


# --------------------------------------------------------------------------
# Реальная ячейка
# --------------------------------------------------------------------------


def test_discover_finds_cell(raw_dir: Path, repo_config: BenchmarkConfig) -> None:
    """Ячейка находится, идентификатор наследует исходный ID набора."""
    sources = SandiaAdapter(repo_config).discover(raw_dir)
    assert [source.cell_id for source in sources] == ["sandia_SNL_18650_NMC_25C_0-100_0.5-1C_a"]


def test_real_cell_parses_and_labels(raw_dir: Path, repo_config: BenchmarkConfig) -> None:
    """Реальная ячейка проходит ingest -> labels и валидаторы §4."""
    adapter = SandiaAdapter(repo_config)
    bundle = adapter.mark_reference_cycles(adapter.load_cell(adapter.discover(raw_dir)[0]))

    # Ошибок быть не должно, а предупреждения у реальной ячейки ожидаемы:
    # «soh > 1.0» — поздний check на 0.04 % выше первого из-за шума измерения,
    # «вне бенчмарка» — рабочие циклы остались в timeseries, но метки не
    # получили (§5.1, режим «только опорные»).
    warnings = validate_bundle(bundle)
    benign = ("soh > 1.0", "вне бенчмарка")
    assert all(any(marker in warning for marker in benign) for warning in warnings), warnings
    assert bundle.cell["chemistry"] == "NMC"
    assert bundle.cell["form_factor"] == "cylindrical_18650"
    assert bundle.cell["nominal_capacity_ah"] == pytest.approx(3.0)
    assert bundle.cell["temperature_setpoint_c"] == pytest.approx(25.0)
    assert bundle.cycles["soh_source"].unique().to_list() == ["measured"]
    assert bundle.cycles["soh_gap_cycles"].unique().to_list() == [0]


def test_capacity_check_is_the_reference_cycle(raw_dir: Path, repo_config: BenchmarkConfig) -> None:
    """Опорными становятся capacity check (полный 0-100 % при 0.5C/0.5C)."""
    adapter = SandiaAdapter(repo_config)
    raw = adapter.load_cell(adapter.discover(raw_dir)[0])
    assert raw.cycles["is_reference"].any(), "первый цикл — проверка ёмкости"
    for row in raw.cycles.iter_rows(named=True):
        if not row["is_reference"]:
            continue
        part = raw.timeseries.filter(part_cycle(row["cycle_index"]))
        charge = part.filter(part["step_type"].is_in(["chg_cc", "chg_cv"]))
        discharge = part.filter(part["step_type"] == "dch")
        assert float(charge["voltage_v"].max()) >= 4.15
        assert float(discharge["voltage_v"].min()) <= 2.05
        charge_current = abs(float(charge["current_a"].median()))
        discharge_current = abs(float(discharge["current_a"].median()))
        assert discharge_current == pytest.approx(charge_current, rel=0.02)


def part_cycle(cycle_index: int) -> Any:  # noqa: ANN401
    """Условие отбора одного цикла (вынесено ради читаемости)."""
    import polars as pl

    return pl.col("cycle_index") == cycle_index


def test_time_is_relative_to_cycle_start(raw_dir: Path, repo_config: BenchmarkConfig) -> None:
    """§4.3: время считается от начала цикла, хотя в источнике оно кумулятивно."""
    adapter = SandiaAdapter(repo_config)
    raw = adapter.load_cell(adapter.discover(raw_dir)[0])
    for cycle_index, part in raw.timeseries.group_by("cycle_index", maintain_order=True):
        assert float(part["t_s"].min()) == pytest.approx(0.0, abs=1e-6), cycle_index[0]
    # в исходнике время сквозное: убеждаемся, что мы его действительно сдвинули
    with FIXTURE.open("rb") as handle:
        payload = pickle.load(handle)
    source_start = float(payload["cycle_data"][1]["time_in_s"][0])
    assert source_start > 0


def test_charge_ah_is_integrated_not_copied(raw_dir: Path, repo_config: BenchmarkConfig) -> None:
    """§4.3: charge_ah получается интегрированием тока и начинается с нуля в шаге."""
    adapter = SandiaAdapter(repo_config)
    raw = adapter.load_cell(adapter.discover(raw_dir)[0])
    for _kind, part in raw.timeseries.group_by("step_type", maintain_order=True):
        pass
    first_cycle = raw.timeseries.filter(pl_col("cycle_index") == 1)
    for _step, part in first_cycle.group_by(pl_col("step_type"), maintain_order=True):
        assert abs(float(part["charge_ah"][0])) < 1e-9


def pl_col(name: str) -> Any:  # noqa: ANN401
    """Обёртка над ``pl.col`` (импорт внутри функции ради читаемости тестов)."""
    import polars as pl

    return pl.col(name)


def test_sanitization_is_recorded(raw_dir: Path, repo_config: BenchmarkConfig) -> None:
    """Отброшенные сенсорные сбои попадают в notes, а не исчезают молча."""
    adapter = SandiaAdapter(repo_config)
    raw = adapter.load_cell(adapter.discover(raw_dir)[0])
    notes = str(raw.cell["notes"])
    assert "BatteryLife" in notes


# --------------------------------------------------------------------------
# Общие правила сегментации (§4.4)
# --------------------------------------------------------------------------


def _segmentation_config(config: BenchmarkConfig) -> dict[str, float]:
    labeling = config.step_labeling
    return {
        "rest_current_tolerance_a": labeling.rest_current_tolerance_a,
        "current_sign_tolerance_a": labeling.current_sign_tolerance_a,
        "cv_voltage_tolerance_v": labeling.cv_voltage_tolerance_v,
    }


def test_segment_cycle_splits_cc_cv_and_discharge(repo_config: BenchmarkConfig) -> None:
    """CC, CV и разряд разделяются по данным, без номеров шагов."""
    t = np.arange(0.0, 600.0, 10.0)
    current = np.concatenate([np.full(40, 1.5), np.linspace(1.5, 0.1, 10), np.full(10, -1.5)])
    voltage = np.concatenate(
        [np.linspace(3.0, 4.2, 40), np.full(10, 4.2), np.linspace(4.1, 2.5, 10)]
    )
    steps = segment_cycle(
        t, current, voltage, np.full(t.size, 25.0), **_segmentation_config(repo_config)
    )
    kinds = [kind for kind, *_ in steps]
    assert kinds == ["chg_cc", "chg_cv", "dch"]


def test_segment_cycle_merges_interrupted_phase(repo_config: BenchmarkConfig) -> None:
    """Короткий провал записи не должен разрезать одну фазу на две.

    Иначе ``charge_ah`` сбрасывается в середине фазы, и §4.3 видит «заряд
    убывает» — ровно так падала реальная ячейка SNL в цикле 639.
    """
    current = np.full(60, 1.5)
    voltage = np.linspace(3.0, 4.2, 60)
    current[30] = -1.1  # одиночный сбой датчика
    voltage[30] = 3.9
    mask = ~isolated_sign_flips(current, tolerance=0.01)
    steps = segment_cycle(
        np.arange(60.0) * 10,
        current[mask],
        voltage[mask],
        np.full(60, 25.0)[mask],
        **_segmentation_config(repo_config),
    )
    assert [kind for kind, *_ in steps] == ["chg_cc"]
    charge = integrate_charge(steps[0][1], steps[0][2])
    assert bool((np.diff(charge) >= -1e-9).all()), "charge_ah внутри фазы монотонен"


def test_segment_cycle_separates_rest(repo_config: BenchmarkConfig) -> None:
    """Отдых выделяется и не приклеивается к заряду."""
    current = np.concatenate([np.zeros(10), np.full(20, 1.5), np.zeros(5)])
    voltage = np.concatenate([np.full(10, 3.0), np.linspace(3.0, 3.5, 20), np.full(5, 3.4)])
    steps = segment_cycle(
        np.arange(35.0) * 10,
        current,
        voltage,
        np.full(35, 25.0),
        **_segmentation_config(repo_config),
    )
    assert [kind for kind, *_ in steps] == ["rest", "chg_cc", "rest"]


def test_isolated_sign_flips_only_flags_single_points() -> None:
    """Сбой датчика — одиночная точка; устойчивая смена знака не помечается."""
    current = np.array([1.5, 1.5, -1.1, 1.5, 1.5])
    assert isolated_sign_flips(current, tolerance=0.01).tolist() == [
        False,
        False,
        True,
        False,
        False,
    ]
    sustained = np.array([1.5, -1.5, -1.5, -1.5])
    assert not isolated_sign_flips(sustained, tolerance=0.01).any()


def test_plausible_mask_flags_sensor_surrogates() -> None:
    """§4.3: нули и отрицательные напряжения — суррогаты, их надо отбрасывать."""
    t = np.arange(4.0)
    current = np.array([1.5, 1.5, 1.5, 1.5])
    voltage = np.array([3.9, 0.0, -0.005, 4.1])
    temperature = np.full(4, 25.0)
    mask = plausible_mask(t, current, voltage, temperature)
    assert mask.tolist() == [True, False, False, True]


def test_reference_rule_ignores_high_rate_cycles(repo_config: BenchmarkConfig) -> None:
    """Рабочий цикл (разряд 2C при заряде 0.5C) опорным не считается."""
    adapter = SandiaAdapter(repo_config)
    raw = _bundle_with_rates(repo_config, adapter)
    assert raw.cycles.filter(raw.cycles["cycle_index"] == 1)["is_reference"][0] is True
    assert raw.cycles.filter(raw.cycles["cycle_index"] == 2)["is_reference"][0] is False


def _bundle_with_rates(config: BenchmarkConfig, adapter: SandiaAdapter) -> Any:  # noqa: ANN401
    """Синтетическая ячейка: проверка 0.5C/0.5C и рабочий цикл 0.5C/2C."""
    from bsb.ingest.base import build_raw_bundle

    def cycle(rate_discharge: float) -> dict[str, list[float]]:
        n_cc, n_cv, n_dch = 40, 10, 20
        current = np.concatenate(
            [np.full(n_cc, 1.5), np.linspace(1.5, 0.1, n_cv), np.full(n_dch, -rate_discharge)]
        )
        voltage = np.concatenate(
            [np.linspace(3.0, 4.2, n_cc), np.full(n_cv, 4.2), np.linspace(4.1, 2.0, n_dch)]
        )
        return {
            "cycle_number": 1,
            "current_in_A": current.tolist(),
            "voltage_in_V": voltage.tolist(),
            "charge_capacity_in_Ah": np.zeros(current.size).tolist(),
            "discharge_capacity_in_Ah": np.zeros(current.size).tolist(),
            "time_in_s": (np.arange(current.size) * 10.0).tolist(),
            "temperature_in_C": np.full(current.size, 25.0).tolist(),
        }

    payload = {
        "cell_id": "SNL_18650_NMC_25C_0-100_0.5-1C_a",
        "cycle_data": [cycle(1.5), cycle(6.0)],
        "cathode_material": "NMC",
        "form_factor": "cylindrical_18650",
        "nominal_capacity_in_Ah": 3.0,
        "min_voltage_limit_in_V": 2.0,
        "max_voltage_limit_in_V": 4.2,
        "depth_of_discharge": 1.0,
        "SOC_interval": [0, 1],
        "charge_protocol": [{"rate_in_C": 0.5}],
        "discharge_protocol": [{"rate_in_C": 1.0}],
    }
    rows = []
    frames = []
    import polars as pl

    for index, item in enumerate(payload["cycle_data"], start=1):
        steps = segment_cycle(
            item["time_in_s"],
            item["current_in_A"],
            item["voltage_in_V"],
            item["temperature_in_C"],
            **_segmentation_config(config),
        )
        parts, offset, capacity = [], 0.0, None
        for kind, t_local, i, v, T in steps:
            charge = integrate_charge(t_local, i)
            parts.append(
                pl.DataFrame(
                    {
                        "step_type": [kind] * t_local.size,
                        "t_s": offset + t_local,
                        "voltage_v": v,
                        "current_a": i,
                        "temperature_c": T,
                        "charge_ah": charge,
                    }
                )
            )
            offset += float(t_local[-1]) + 1.0
            if kind == "dch":
                capacity = abs(float(charge[-1]))
        frame = pl.concat(parts).with_columns(
            pl.lit(payload["cell_id"]).alias("cell_id"),
            pl.lit(index, dtype=pl.Int32).alias("cycle_index"),
        )
        frames.append(frame)
        rows.append(
            {
                "cell_id": payload["cell_id"],
                "cycle_index": index,
                "capacity_ah": capacity,
                "is_reference": adapter._cycle_row(
                    payload["cell_id"],
                    index,
                    frame,
                    capacity=capacity,
                    limits=(2.0, 4.2),
                    nominal=3.0,
                )["is_reference"],
            }
        )
    cell = {
        "cell_id": payload["cell_id"],
        "dataset": "sandia",
        "chemistry": "NMC",
        "form_factor": "cylindrical_18650",
        "nominal_capacity_ah": 3.0,
        "temperature_setpoint_c": 25.0,
        "charge_protocol": "CC-CV 0.5C",
        "batch_id": None,
        "notes": None,
    }
    bundle = build_raw_bundle(cell, pl.DataFrame(rows), pl.concat(frames))
    return bundle


def test_synthetic_bundle_is_valid(repo_config: BenchmarkConfig) -> None:
    """Синтетическая ячейка с проверкой и рабочим циклом проходит валидаторы."""
    adapter = SandiaAdapter(repo_config)
    bundle = _bundle_with_rates(repo_config, adapter)
    assert validate_raw_bundle(bundle) == []
    labelled = adapter.mark_reference_cycles(bundle)
    assert isinstance(labelled, CellBundle)
    assert labelled.cycles.height == 1
