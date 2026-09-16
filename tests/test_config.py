"""Тесты конфига: единственный источник правды обязан быть строгим."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

from bsb.config import BenchmarkConfig, ConfigError, load_config, load_config_from_string

REPO_ROOT = Path(__file__).resolve().parents[1]


def _raw_config() -> dict[str, Any]:
    """Разобранный репозиторный YAML (мутируемая копия)."""
    return yaml.safe_load((REPO_ROOT / "configs" / "benchmark.yaml").read_text(encoding="utf-8"))


def _load(data: dict[str, Any], tmp_path: Path) -> BenchmarkConfig:
    """Собрать конфиг из словаря с корнем в ``tmp_path``."""
    return load_config_from_string(yaml.safe_dump(data), repo_root=tmp_path)


# --------------------------------------------------------------------------
# Репозиторный конфиг
# --------------------------------------------------------------------------


def test_repo_config_matches_owner_decisions(repo_config: BenchmarkConfig) -> None:
    """Зафиксированные решения владельца отражены в конфиге."""
    assert repo_config.labels.reference_only is True
    assert repo_config.labels.allow_interpolated is False
    assert repo_config.labels.allow_extrapolation is False
    assert repo_config.eval.main_test_chemistries == ("NCA", "NMC")
    assert repo_config.eval.hard_mode_chemistries == ("LFP",)
    assert repo_config.splits.test_datasets == ("sandia",)
    assert repo_config.splits.unit == "cell"


def test_repo_config_fragment_defaults(repo_config: BenchmarkConfig) -> None:
    """Параметры фрагментов из §7 на месте."""
    fragment = repo_config.fragment
    assert fragment.delta_v == 0.2
    assert fragment.n_points == 128
    assert fragment.phase == "chg_cc"
    assert fragment.v_normalization == "per_cell_range"
    assert fragment.channels == ("q_norm", "current", "temperature")
    assert fragment.min_points_in_window == 20
    assert len(fragment.delta_v_grid) == 4


def test_paths_resolve_against_repo_root(repo_config: BenchmarkConfig) -> None:
    """Относительные пути резолвятся от корня репозитория."""
    assert repo_config.paths.unified_cells == REPO_ROOT / "data" / "unified" / "cells.parquet"
    assert repo_config.paths.timeseries_path("oxford_Cell1") == (
        REPO_ROOT / "data" / "unified" / "timeseries" / "oxford_Cell1.parquet"
    )


def test_fragments_path_hides_test_labels(repo_config: BenchmarkConfig) -> None:
    """Тестовый файл участников — без меток, обучающие — с метками (ТЗ §7)."""
    assert repo_config.paths.fragments_path("train").name == "train.parquet"
    assert repo_config.paths.fragments_path("val").name == "val.parquet"
    assert repo_config.paths.fragments_path("test").name == "test_unlabeled.parquet"
    assert repo_config.paths.fragments_path("test", labeled=True).name == "test.parquet"


def test_make_config_relocates_paths(
    make_config: Callable[[dict[str, Any] | None], BenchmarkConfig], tmp_path: Path
) -> None:
    """Фабрика конфигов переносит пути во временный каталог."""
    config = make_config({"fragment.delta_v": 0.5})
    assert config.fragment.delta_v == 0.5
    assert config.paths.data_dir == tmp_path / "data"
    assert config.root == tmp_path


# --------------------------------------------------------------------------
# Строгость
# --------------------------------------------------------------------------


def test_unknown_top_level_key_is_rejected(tmp_path: Path) -> None:
    """Опечатка в имени секции не должна молча игнорироваться."""
    data = _raw_config()
    data["fragments"] = data.pop("fragment")
    with pytest.raises(ConfigError, match="неизвестные ключи"):
        _load(data, tmp_path)


def test_unknown_nested_key_is_rejected(tmp_path: Path) -> None:
    """То же для вложенного ключа."""
    data = _raw_config()
    data["fragment"]["delta_v_volts"] = 0.2
    with pytest.raises(ConfigError, match="неизвестные ключи"):
        _load(data, tmp_path)


def test_missing_section_is_rejected(tmp_path: Path) -> None:
    """Пропавшая секция — ошибка."""
    data = _raw_config()
    del data["labels"]
    with pytest.raises(ConfigError, match="отсутствуют обязательные ключи"):
        _load(data, tmp_path)


def test_missing_file_is_reported(tmp_path: Path) -> None:
    """Несуществующий конфиг даёт внятную ошибку."""
    with pytest.raises(ConfigError, match="не найден"):
        load_config(tmp_path / "nope.yaml")


def test_non_mapping_root_is_rejected(tmp_path: Path) -> None:
    """Список вместо словаря на верхнем уровне — ошибка."""
    with pytest.raises(ConfigError, match="словарь верхнего уровня"):
        load_config_from_string("- 1\n- 2\n", repo_root=tmp_path)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"version": "1"}, "version"),
        ({"seed": 1.5}, "seed"),
        ({"splits.unit": "cycle"}, "splits.unit"),
        ({"labels.allow_extrapolation": True}, "экстраполяция"),
        ({"labels.max_soh_gap_cycles": 0}, "max_soh_gap_cycles"),
        ({"fragment.delta_v": 0.0}, "delta_v"),
        ({"fragment.n_points": 1}, "n_points"),
        ({"fragment.phase": "dch"}, "phase"),
        ({"fragment.v_normalization": "zscore"}, "v_normalization"),
        ({"fragment.channels": ["current"]}, "q_norm"),
        ({"fragment.channels": ["q_norm", "voltage"]}, "неизвестные каналы"),
        ({"fragment.delta_v_grid": []}, "delta_v_grid"),
        ({"step_labeling.min_step_points": 0}, "min_step_points"),
        ({"ingest.kit.target_resolution_s": 0}, "target_resolution_s"),
        ({"eval.prediction_bounds": [1.0, 1.0]}, "prediction_bounds"),
        ({"eval.calibration_coverage": 1.5}, "calibration_coverage"),
        ({"eval.soh_deciles": 1}, "soh_deciles"),
        ({"noise_floor.min_replicates": 1}, "min_replicates"),
        ({"leakage_probe.max_relative_improvement": -0.1}, "max_relative_improvement"),
        ({"baseline.savgol.window_length": 10}, "window_length"),
        ({"baseline.savgol.polyorder": 11}, "polyorder"),
        ({"baseline.elastic_net.l1_ratio": 1.5}, "l1_ratio"),
        ({"paths.unified_dir": 5}, "unified_dir"),
    ],
)
def test_bad_values_are_rejected(tmp_path: Path, override: dict[str, Any], message: str) -> None:
    """Плохое значение в любой секции — ошибка с указанием ключа."""
    data = _raw_config()
    for dotted, value in override.items():
        node = data
        *parents, leaf = dotted.split(".")
        for key in parents:
            node = node[key]
        node[leaf] = value
    with pytest.raises(ConfigError, match=message):
        _load(data, tmp_path)


def test_interpolation_requires_working_cycles(tmp_path: Path) -> None:
    """§5.1: интерполяция поверх опорных циклов смысла не имеет."""
    data = _raw_config()
    data["labels"]["allow_interpolated"] = True
    with pytest.raises(ConfigError, match="reference_only=false"):
        _load(data, tmp_path)


def test_interpolation_flag_is_allowed_with_working_cycles(tmp_path: Path) -> None:
    """Опциональный режим §5.1 включается только вместе с reference_only=false."""
    data = _raw_config()
    data["labels"]["reference_only"] = False
    data["labels"]["allow_interpolated"] = True
    config = _load(data, tmp_path)
    assert config.labels.allow_interpolated is True
    assert config.labels.reference_only is False
