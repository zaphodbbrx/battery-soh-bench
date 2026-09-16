"""Тесты генератора фрагментов (ТЗ §7, §14)."""

from __future__ import annotations

import io
from collections.abc import Callable
from typing import Any

import numpy as np
import polars as pl
import pytest

from bsb.config import BenchmarkConfig
from bsb.dataset import sample_fragments
from bsb.fragments import (
    fragment_id,
    generate_fragments,
    iter_fragment_windows,
    iter_sampled_windows,
    strip_labels,
)
from bsb.ingest.base import RawBundle
from bsb.labels import build_labels
from bsb.schema import CellBundle, validate_fragments
from synthetic import synth_bundle


def _frame_bytes(frame: pl.DataFrame) -> bytes:
    """Байтовое представление кадра — для проверки побитового детерминизма."""
    buffer = io.BytesIO()
    frame.write_parquet(buffer)
    return buffer.getvalue()


# --------------------------------------------------------------------------
# Детерминизм §14
# --------------------------------------------------------------------------


def test_two_generations_are_bitwise_identical(
    synthetic_bundle: CellBundle, repo_config: BenchmarkConfig
) -> None:
    """Две генерации с одним seed совпадают побитово (ТЗ §14)."""
    first = generate_fragments(synthetic_bundle, repo_config)
    second = generate_fragments(synthetic_bundle, repo_config)
    assert first.equals(second)
    assert _frame_bytes(first) == _frame_bytes(second)


def test_generation_does_not_depend_on_cell_order(
    make_config: Callable[[dict[str, Any] | None], BenchmarkConfig],
) -> None:
    """Порядок обхода ячеек не влияет на результат: ГПСЧ привязан к окну."""
    config = make_config()
    left = generate_fragments(synth_bundle(cell_id="oxford_Cell1"), config)
    other = generate_fragments(synth_bundle(cell_id="oxford_Cell2"), config)
    combined = pl.concat([other, left], how="vertical")
    assert left.equals(combined.filter(pl.col("cell_id") == "oxford_Cell1"))


def test_fragment_id_is_a_hash_of_identity_and_seed() -> None:
    """Идентификатор фрагмента — хеш от (cell_id, cycle_index, v_start, seed) §7."""
    base = fragment_id("oxford_Cell1", 101, 3.9, 20260901)
    assert base == fragment_id("oxford_Cell1", 101, 3.9, 20260901)
    assert base != fragment_id("oxford_Cell1", 101, 3.9001, 20260901)
    assert base != fragment_id("oxford_Cell1", 102, 3.9, 20260901)
    assert base != fragment_id("oxford_Cell2", 101, 3.9, 20260901)
    assert base != fragment_id("oxford_Cell1", 101, 3.9, 1)
    assert len(base) == 16


def test_fragment_ids_are_unique_within_split(
    synthetic_bundle: CellBundle, repo_config: BenchmarkConfig
) -> None:
    """Коллизий идентификаторов нет — иначе сабмит не сходится с эталоном."""
    fragments = generate_fragments(synthetic_bundle, repo_config)
    assert fragments["fragment_id"].n_unique() == fragments.height


# --------------------------------------------------------------------------
# Алгоритм §7
# --------------------------------------------------------------------------


def test_window_width_and_grid(synthetic_bundle: CellBundle, repo_config: BenchmarkConfig) -> None:
    """Окно имеет ровно ``delta_v`` ширины и ``n_points`` точек в каждом канале."""
    fragments = generate_fragments(synthetic_bundle, repo_config)
    width = fragments["v_end"] - fragments["v_start"]
    assert width.to_list() == pytest.approx([repo_config.fragment.delta_v] * fragments.height)
    for channel in ("q_norm", "current", "temperature"):
        assert fragments[channel].list.len().unique().to_list() == [repo_config.fragment.n_points]


def test_one_fragment_per_labelled_cycle(
    synthetic_bundle: CellBundle, repo_config: BenchmarkConfig
) -> None:
    """По умолчанию одно окно на размеченный цикл."""
    fragments = generate_fragments(synthetic_bundle, repo_config)
    assert fragments.height == synthetic_bundle.cycles.height


def test_windows_per_cycle_is_configurable(
    synthetic_bundle: CellBundle,
    make_config: Callable[[dict[str, Any] | None], BenchmarkConfig],
) -> None:
    """Число окон на цикл берётся из конфига."""
    config = make_config({"fragment.windows_per_cycle": 3})
    fragments = generate_fragments(synthetic_bundle, config)
    assert fragments.height == synthetic_bundle.cycles.height * 3
    assert fragments["fragment_id"].n_unique() == fragments.height


def test_windows_stay_inside_the_charge_phase(
    synthetic_bundle: CellBundle, repo_config: BenchmarkConfig
) -> None:
    """Окно целиком лежит внутри CC-фазы заряда, а не за её границами."""
    charge = synthetic_bundle.timeseries.filter(pl.col("step_type") == "chg_cc")
    low = float(charge["voltage_v"].min())
    high = float(charge["voltage_v"].max())
    fragments = generate_fragments(synthetic_bundle, repo_config)
    assert fragments["v_start"].min() >= low
    assert fragments["v_end"].max() <= high + 1e-9


def test_q_norm_is_normalized_by_this_cell_bol(
    synthetic_bundle: CellBundle, repo_config: BenchmarkConfig
) -> None:
    """§7 п.4: Q нормируется на ``bol`` этой ячейки, и её можно восстановить."""
    fragments = generate_fragments(synthetic_bundle, repo_config)
    bol = float(synthetic_bundle.cell["bol_capacity_ah"])
    charge = synthetic_bundle.timeseries.filter(pl.col("step_type") == "chg_cc")
    for row in fragments.head(3).iter_rows(named=True):
        phase = charge.filter(pl.col("cycle_index") == row["cycle_index"]).sort("voltage_v")
        voltage = phase["voltage_v"].to_numpy()
        charge_ah = phase["charge_ah"].to_numpy()
        # q_norm[k] = Q(v_k)/bol на равномерной сетке окна.
        grid = np.linspace(row["v_start"], row["v_end"], len(row["q_norm"]))
        assert row["q_norm"] == pytest.approx(
            (np.interp(grid, voltage, charge_ah) / bol).tolist(), rel=1e-12
        )
        # Значит, bol восстанавливается из отношения заряда к q_norm.
        recovered = float(np.interp(row["v_end"], voltage, charge_ah)) / row["q_norm"][-1]
        assert recovered == pytest.approx(bol, rel=1e-9)


def test_current_channel_is_present(
    synthetic_bundle: CellBundle, repo_config: BenchmarkConfig
) -> None:
    """Канал тока обязателен: без него модель путает сдвиг по току со старением (§17.3)."""
    fragments = generate_fragments(synthetic_bundle, repo_config)
    assert fragments["current"].list.mean().min() == pytest.approx(0.36, rel=1e-6)


def test_min_points_in_window_drops_sparse_windows(
    synthetic_bundle: CellBundle,
    make_config: Callable[[dict[str, Any] | None], BenchmarkConfig],
) -> None:
    """Мало точек в окне — фрагмент отбрасывается, а не интерполируется наугад."""
    config = make_config({"fragment.min_points_in_window": 10_000})
    fragments = generate_fragments(synthetic_bundle, config)
    assert fragments.height == 0
    assert fragments.columns == list(generate_fragments(synthetic_bundle, make_config()).columns)


def test_non_reference_cycles_never_reach_fragments(
    make_config: Callable[[dict[str, Any] | None], BenchmarkConfig],
) -> None:
    """§5.1: циклы без метки в бенчмарк не попадают, хотя timeseries их хранит."""
    config = make_config()
    raw = synth_bundle()
    with_unlabelled = synth_bundle(non_reference_cycles=(3,))
    fragments = generate_fragments(with_unlabelled, config)
    assert 3 not in fragments["cycle_index"].to_list()
    assert raw.cycles.height == fragments.height + 1


# --------------------------------------------------------------------------
# v_normalization §9
# --------------------------------------------------------------------------


def test_per_cell_range_uses_the_cells_own_span(
    make_config: Callable[[dict[str, Any] | None], BenchmarkConfig],
) -> None:
    """§9: per_cell_range даёт окна на любой химии, включая узкий диапазон LFP."""
    config = make_config({"fragment.v_normalization": "per_cell_range"})
    lfp_like = synth_bundle(cell_id="sandia_LFP1", dataset="sandia", chemistry="LFP")
    narrow = lfp_like.timeseries.with_columns((pl.col("voltage_v") * 0.2 + 2.8).alias("voltage_v"))
    bundle = CellBundle(cells=lfp_like.cells, cycles=lfp_like.cycles, timeseries=narrow)
    fragments = generate_fragments(bundle, config)
    assert fragments.height == lfp_like.cycles.height


def test_absolute_bounds_outside_cell_span_yield_nothing(
    make_config: Callable[[dict[str, Any] | None], BenchmarkConfig],
) -> None:
    """§9: при absolute окно выше рабочего диапазона LFP не существует физически."""
    config = make_config(
        {"fragment.v_normalization": "absolute", "fragment.absolute_bounds": [4.6, 4.8]}
    )
    fragments = generate_fragments(synth_bundle(), config)
    assert fragments.height == 0


def test_absolute_bounds_inside_cell_span_work(
    make_config: Callable[[dict[str, Any] | None], BenchmarkConfig],
) -> None:
    """При absolute внутри рабочего диапазона окна строятся как обычно."""
    config = make_config(
        {"fragment.v_normalization": "absolute", "fragment.absolute_bounds": [3.4, 3.9]}
    )
    fragments = generate_fragments(synth_bundle(), config)
    assert fragments.height > 0
    assert fragments["v_start"].min() >= 3.4
    assert fragments["v_end"].max() <= 3.9001


# --------------------------------------------------------------------------
# Режим sampling §7
# --------------------------------------------------------------------------


def test_sampling_is_random_but_seeded_by_the_caller(
    synthetic_bundle: CellBundle, repo_config: BenchmarkConfig
) -> None:
    """sampling не воспроизводим по fragment.seed, но воспроизводим по своему rng."""
    first = iter_sampled_windows(synthetic_bundle, repo_config, np.random.default_rng(1), windows=2)
    same = iter_sampled_windows(synthetic_bundle, repo_config, np.random.default_rng(1), windows=2)
    other = iter_sampled_windows(synthetic_bundle, repo_config, np.random.default_rng(2), windows=2)
    assert [w.v_start for w in first] == [w.v_start for w in same]
    assert [w.v_start for w in first] != [w.v_start for w in other]
    fixed = {w.v_start for w in iter_fragment_windows(synthetic_bundle, repo_config)}
    assert all(w.v_start not in fixed for w in first)


def test_sample_fragments_returns_labeled_rows(
    synthetic_bundle: CellBundle, repo_config: BenchmarkConfig
) -> None:
    """Участнический доступ к аугментации отдаёт те же каналы и метку SOH."""
    frame = sample_fragments(
        [synthetic_bundle], repo_config, windows=2, rng=np.random.default_rng(0)
    )
    assert frame.height == synthetic_bundle.cycles.height * 2
    validate_fragments(frame, labeled=True, n_points=repo_config.fragment.n_points)


def test_sample_fragments_can_hide_labels(
    synthetic_bundle: CellBundle, repo_config: BenchmarkConfig
) -> None:
    """Без меток остаются только входы — как в тестовом файле участников."""
    frame = sample_fragments(
        [synthetic_bundle], repo_config, windows=1, rng=np.random.default_rng(0), labeled=False
    )
    assert frame.columns == ["fragment_id", "v_start", "v_end", "q_norm", "current", "temperature"]
    validate_fragments(frame, labeled=False, n_points=repo_config.fragment.n_points)


# --------------------------------------------------------------------------
# Тестовый файл участников §7
# --------------------------------------------------------------------------


def test_strip_labels_matches_participant_schema(
    synthetic_bundle: CellBundle, repo_config: BenchmarkConfig
) -> None:
    """Тестовый файл участников не содержит ни ячейки, ни цикла, ни SOH (§7)."""
    fragments = generate_fragments(synthetic_bundle, repo_config)
    stripped = strip_labels(fragments)
    assert "soh" not in stripped.columns
    assert "cell_id" not in stripped.columns
    assert "cycle_index" not in stripped.columns
    assert stripped.height == fragments.height
    validate_fragments(stripped, labeled=False, n_points=repo_config.fragment.n_points)


# --------------------------------------------------------------------------
# Round-trip §14
# --------------------------------------------------------------------------


def test_roundtrip_recovers_known_capacity(
    repo_config: BenchmarkConfig,
) -> None:
    """§14: синтетическая ячейка с известной ёмкостью проходит путь до фрагментов.

    Из фрагмента восстанавливаются и ``bol`` (через нормировку ``q_norm``), и
    целевая переменная SOH цикла.
    """
    known_bol = 0.72
    raw: RawBundle = synth_bundle(bol_capacity_ah=known_bol)
    bundle = build_labels(raw, repo_config)
    fragments = generate_fragments(bundle, repo_config)
    assert fragments.height == bundle.cycles.height

    # SOH во фрагменте совпадает с SOH размеченного цикла.
    joined = fragments.join(
        bundle.cycles.select("cycle_index", "soh"), on="cycle_index", suffix="_label"
    )
    assert joined["soh"].to_list() == pytest.approx(joined["soh_label"].to_list())

    # bol восстанавливается из отношения заряда к q_norm.
    charge = bundle.timeseries.filter(pl.col("step_type") == "chg_cc")
    row = fragments.row(0, named=True)
    phase = charge.filter(pl.col("cycle_index") == row["cycle_index"]).sort("voltage_v")
    total = float(
        np.interp(row["v_end"], phase["voltage_v"].to_numpy(), phase["charge_ah"].to_numpy())
    )
    assert total / row["q_norm"][-1] == pytest.approx(known_bol, rel=1e-9)
