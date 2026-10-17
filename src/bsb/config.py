"""Минимальная публичная конфигурация оценщика и примера признаков."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_NAME = "configs/benchmark.yaml"


class ConfigError(ValueError):
    """Конфиг оценщика некорректен."""


@dataclass(frozen=True, slots=True)
class PathsConfig:
    """Пути к публичным parquet и результатам."""

    repo_root: Path
    fragments_dir: Path
    results_path: Path

    def fragments_path(self, split: str, *, labeled: bool | None = None) -> Path:
        """Путь к parquet-фрагментам; test по умолчанию без разметки."""
        if split not in {"train", "val", "test"}:
            raise ConfigError(f"неизвестный split: {split!r}")
        if labeled is None:
            labeled = split != "test"
        suffix = "" if labeled else "_unlabeled"
        return self.fragments_dir / f"{split}{suffix}.parquet"


@dataclass(frozen=True, slots=True)
class EvalConfig:
    """Параметры публичного scoring-контракта."""

    main_test_chemistries: tuple[str, ...]
    hard_mode_chemistries: tuple[str, ...]
    calibration_coverage: float
    soh_deciles: int
    prediction_bounds: tuple[float, float]


@dataclass(frozen=True, slots=True)
class BaselineConfig:
    """Настройки сглаживания, используемые примером извлечения признаков."""

    model: str
    savgol_window_length: int
    savgol_polyorder: int
    elastic_net_alpha: float
    elastic_net_l1_ratio: float
    elastic_net_max_iter: int


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    """Настройки оценивания и общие публичные пути."""

    seed: int
    source_path: Path
    paths: PathsConfig
    eval: EvalConfig
    baseline: BaselineConfig

    @property
    def root(self) -> Path:
        """Корень публичного репозитория."""
        return self.paths.repo_root


_LOAD_CACHE: dict[Path, BenchmarkConfig] = {}


def load_config(path: str | Path | None = None) -> BenchmarkConfig:
    """Загрузить и проверить минимальный конфиг scoring-контракта."""
    source = Path(path) if path is not None else Path(DEFAULT_CONFIG_NAME)
    if not source.is_absolute():
        source = Path.cwd() / source
    source = source.resolve()
    if source in _LOAD_CACHE:
        return _LOAD_CACHE[source]
    if not source.is_file():
        raise FileNotFoundError(f"конфиг не найден: {source}")
    raw: Any = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ConfigError("корень конфига должен быть mapping")
    required = {"seed", "paths", "eval", "baseline"}
    if set(raw) != required:
        raise ConfigError(f"ожидались ключи {sorted(required)}, получено {sorted(raw)}")
    root = source.parent.parent

    def section(name: str, keys: set[str]) -> dict[str, Any]:
        value = raw[name]
        if not isinstance(value, dict) or set(value) != keys:
            raise ConfigError(f"{name}: ожидались ключи {sorted(keys)}")
        return value

    paths = section("paths", {"fragments_dir", "results_path"})
    evaluation = section(
        "eval",
        {
            "main_test_chemistries",
            "hard_mode_chemistries",
            "calibration_coverage",
            "soh_deciles",
            "prediction_bounds",
        },
    )
    baseline = section(
        "baseline",
        {
            "model",
            "savgol_window_length",
            "savgol_polyorder",
            "elastic_net_alpha",
            "elastic_net_l1_ratio",
            "elastic_net_max_iter",
        },
    )
    try:
        config = BenchmarkConfig(
            seed=int(raw["seed"]),
            source_path=source,
            paths=PathsConfig(
                repo_root=root,
                fragments_dir=(root / str(paths["fragments_dir"])).resolve(),
                results_path=(root / str(paths["results_path"])).resolve(),
            ),
            eval=EvalConfig(
                main_test_chemistries=tuple(evaluation["main_test_chemistries"]),
                hard_mode_chemistries=tuple(evaluation["hard_mode_chemistries"]),
                calibration_coverage=float(evaluation["calibration_coverage"]),
                soh_deciles=int(evaluation["soh_deciles"]),
                prediction_bounds=tuple(float(x) for x in evaluation["prediction_bounds"]),
            ),
            baseline=BaselineConfig(
                model=str(baseline["model"]),
                savgol_window_length=int(baseline["savgol_window_length"]),
                savgol_polyorder=int(baseline["savgol_polyorder"]),
                elastic_net_alpha=float(baseline["elastic_net_alpha"]),
                elastic_net_l1_ratio=float(baseline["elastic_net_l1_ratio"]),
                elastic_net_max_iter=int(baseline["elastic_net_max_iter"]),
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfigError(f"ошибка параметров конфига: {exc}") from exc
    if len(config.eval.prediction_bounds) != 2:
        raise ConfigError("eval.prediction_bounds должен содержать два числа")
    _LOAD_CACHE[source] = config
    return config
