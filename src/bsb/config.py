"""Загрузка ``configs/benchmark.yaml`` — единственного источника правды (ТЗ §3).

Модуль намеренно строгий: неизвестный ключ, отсутствующий ключ или значение
неверного типа — это ошибка, а не молчаливый дефолт. Так параметр, добавленный
в YAML и забытый в коде (или наоборот), не может тихо разойтись с реальностью.

Пример
------
>>> cfg = load_config()                      # configs/benchmark.yaml по умолчанию
>>> cfg.fragment.delta_v, cfg.fragment.n_points
(0.2, 128)
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Final, Literal

import yaml

__all__ = [
    "BaselineConfig",
    "BenchmarkConfig",
    "CalceConfig",
    "ConfigError",
    "EvalConfig",
    "FragmentConfig",
    "IngestConfig",
    "KitConfig",
    "LabelsConfig",
    "LeakageProbeConfig",
    "NoiseFloorConfig",
    "PathsConfig",
    "SplitsConfig",
    "StepLabelingConfig",
    "load_config",
]

DEFAULT_CONFIG_NAME: Final[str] = "configs/benchmark.yaml"
V_NORMALIZATIONS: Final[tuple[str, ...]] = ("absolute", "per_cell_range")
FRAGMENT_CHANNELS: Final[tuple[str, ...]] = ("q_norm", "current", "temperature")
SPLIT_UNITS: Final[tuple[str, ...]] = ("cell", "batch")


class ConfigError(ValueError):
    """Конфиг не соответствует контракту: неизвестный ключ или плохое значение."""


class _Missing:
    """Sentinel «ключа нет» (отличается от ``None``)."""


_MISSING: Final = _Missing()


# --------------------------------------------------------------------------
# Помощники разбора
# --------------------------------------------------------------------------


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    """Проверить, что узел — отображение."""
    if not isinstance(value, Mapping):
        raise ConfigError(f"{path}: ожидался словарь, получено {type(value).__name__}")
    return value


def _check_keys(
    data: Mapping[str, Any], path: str, required: Sequence[str], optional: Sequence[str] = ()
) -> None:
    """Проверить ровно ожидаемый набор ключей."""
    unknown = sorted(set(data) - set(required) - set(optional))
    if unknown:
        raise ConfigError(
            f"{path}: неизвестные ключи {unknown}; "
            f"допустимые: {sorted(set(required) | set(optional))}"
        )
    missing = [key for key in required if key not in data]
    if missing:
        raise ConfigError(f"{path}: отсутствуют обязательные ключи {missing}")


def _get(data: Mapping[str, Any], key: str, path: str, default: Any = _MISSING) -> Any:
    """Достать значение или дефолт, иначе ошибка."""
    if key in data:
        return data[key]
    if isinstance(default, _Missing):
        raise ConfigError(f"{path}.{key}: обязательный ключ отсутствует")
    return default


def _as_bool(value: Any, path: str) -> bool:
    """Строгий bool (``1`` и ``"true"`` не принимаются)."""
    if not isinstance(value, bool):
        raise ConfigError(f"{path}: ожидался bool, получено {value!r}")
    return value


def _as_int(value: Any, path: str) -> int:
    """Строгий int (bool не считается числом)."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{path}: ожидался int, получено {value!r}")
    return value


def _as_float(value: Any, path: str) -> float:
    """Строгий float, принимает и int."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigError(f"{path}: ожидался float, получено {value!r}")
    return float(value)


def _as_str(value: Any, path: str) -> str:
    """Строгий str."""
    if not isinstance(value, str):
        raise ConfigError(f"{path}: ожидался str, получено {value!r}")
    return value


def _as_str_tuple(value: Any, path: str) -> tuple[str, ...]:
    """Список строк в tuple (для неизменяемости конфига)."""
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ConfigError(f"{path}: ожидался список строк, получено {value!r}")
    return tuple(value)


def _as_float_tuple(value: Any, path: str) -> tuple[float, ...]:
    """Список чисел в tuple."""
    if not isinstance(value, list):
        raise ConfigError(f"{path}: ожидался список чисел, получено {value!r}")
    return tuple(_as_float(item, f"{path}[{index}]") for index, item in enumerate(value))


# --------------------------------------------------------------------------
# Секции
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PathsConfig:
    """Пути тулкита. Относительные пути резолвятся от корня репозитория."""

    repo_root: Path
    data_dir: Path
    raw_dir: Path
    unified_dir: Path
    fragments_dir: Path
    splits_dir: Path
    results_path: Path

    @property
    def unified_cells(self) -> Path:
        """``cells.parquet`` унифицированного уровня (ТЗ §4.1)."""
        return self.unified_dir / "cells.parquet"

    @property
    def unified_cycles(self) -> Path:
        """``cycles.parquet`` унифицированного уровня (ТЗ §4.2)."""
        return self.unified_dir / "cycles.parquet"

    @property
    def timeseries_dir(self) -> Path:
        """Каталог ``timeseries/``, партиционированный по ``cell_id`` (ТЗ §4.3)."""
        return self.unified_dir / "timeseries"

    def timeseries_path(self, cell_id: str) -> Path:
        """Путь партиции одной ячейки."""
        return self.timeseries_dir / f"{cell_id}.parquet"

    def fragments_path(self, split: str, *, labeled: bool | None = None) -> Path:
        """Путь файла фрагментов сплита.

        Для ``test`` по умолчанию берётся файл участников без меток (ТЗ §7).
        """
        if labeled is None:
            labeled = split != "test"
        suffix = "" if labeled else "_unlabeled"
        return self.fragments_dir / f"{split}{suffix}.parquet"


@dataclass(frozen=True, slots=True)
class SplitsConfig:
    """Сплиты (ТЗ §8)."""

    unit: Literal["cell", "batch"]
    files: Mapping[str, str]
    train_datasets: tuple[str, ...]
    val_datasets: tuple[str, ...]
    test_datasets: tuple[str, ...]

    @property
    def split_names(self) -> tuple[str, ...]:
        """Имена сплитов в порядке train / val / test."""
        return ("train", "val", "test")

    def filename(self, split: str) -> str:
        """Имя файла со списком ячеек для сплита."""
        if split not in self.files:
            raise ConfigError(f"splits: неизвестный сплит {split!r}; есть {sorted(self.files)}")
        return self.files[split]


@dataclass(frozen=True, slots=True)
class LabelsConfig:
    """Целевая переменная (ТЗ §5, §5.1)."""

    reference_only: bool
    allow_interpolated: bool
    max_soh_gap_cycles: int
    allow_extrapolation: bool
    bol_source: str
    reference_cycle_rules: Mapping[str, Mapping[str, Any]]

    def rule(self, dataset: str) -> Mapping[str, Any]:
        """Правило определения опорного цикла для датасета."""
        if dataset not in self.reference_cycle_rules:
            raise ConfigError(f"labels.reference_cycle_rules: нет правила для датасета {dataset!r}")
        return self.reference_cycle_rules[dataset]


@dataclass(frozen=True, slots=True)
class StepLabelingConfig:
    """Пороги разметки ``step_type`` (ТЗ §4.4)."""

    cc_current_tolerance_rel: float
    cv_voltage_tolerance_v: float
    min_step_points: int
    rest_current_tolerance_a: float
    current_sign_tolerance_a: float


@dataclass(frozen=True, slots=True)
class FragmentConfig:
    """Генератор фрагментов (ТЗ §7)."""

    delta_v: float
    n_points: int
    phase: str
    v_normalization: Literal["absolute", "per_cell_range"]
    channels: tuple[str, ...]
    min_points_in_window: int
    seed: int
    windows_per_cycle: int
    delta_v_grid: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class KitConfig:
    """Параметры ingest KIT (ТЗ §6.3)."""

    target_resolution_s: float
    keep_cc_full_resolution: bool
    include_calendar_aging: bool


@dataclass(frozen=True, slots=True)
class CalceConfig:
    """Параметры ingest CALCE (ТЗ §6.2)."""

    series: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class IngestConfig:
    """Параметры ingest по датасетам (ТЗ §6)."""

    kit: KitConfig
    calce: CalceConfig
    oxford: Mapping[str, Any] = field(default_factory=dict)
    sandia: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EvalConfig:
    """Метрики (ТЗ §10)."""

    main_test_chemistries: tuple[str, ...]
    hard_mode_chemistries: tuple[str, ...]
    calibration_coverage: float
    soh_deciles: int
    prediction_bounds: tuple[float, float]


@dataclass(frozen=True, slots=True)
class LeakageProbeConfig:
    """Проба на утечку (ТЗ §12.3)."""

    max_relative_improvement: float


@dataclass(frozen=True, slots=True)
class NoiseFloorConfig:
    """Шумовой пол (ТЗ §12.4)."""

    min_replicates: int
    replicate_keys: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class BaselineConfig:
    """Бейзлайны (ТЗ §12.5)."""

    model: str
    savgol_window_length: int
    savgol_polyorder: int
    elastic_net_alpha: float
    elastic_net_l1_ratio: float
    elastic_net_max_iter: int


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    """Конфиг бенчмарка целиком."""

    version: int
    seed: int
    source_path: Path
    paths: PathsConfig
    splits: SplitsConfig
    labels: LabelsConfig
    step_labeling: StepLabelingConfig
    fragment: FragmentConfig
    ingest: IngestConfig
    eval: EvalConfig
    leakage_probe: LeakageProbeConfig
    noise_floor: NoiseFloorConfig
    baseline: BaselineConfig

    @property
    def root(self) -> Path:
        """Корень репозитория (от него резолвятся относительные пути)."""
        return self.paths.repo_root

    @classmethod
    def from_dict(
        cls, data: Mapping[str, Any], *, source_path: Path, repo_root: Path
    ) -> BenchmarkConfig:
        """Собрать конфиг из разобранного YAML со строгой проверкой."""
        _check_keys(
            data,
            "config",
            required=(
                "version",
                "seed",
                "paths",
                "splits",
                "labels",
                "step_labeling",
                "fragment",
                "ingest",
                "eval",
                "leakage_probe",
                "noise_floor",
                "baseline",
            ),
        )
        return cls(
            version=_as_int(data["version"], "config.version"),
            seed=_as_int(data["seed"], "config.seed"),
            source_path=source_path,
            paths=_parse_paths(data["paths"], repo_root=repo_root),
            splits=_parse_splits(data["splits"]),
            labels=_parse_labels(data["labels"]),
            step_labeling=_parse_step_labeling(data["step_labeling"]),
            fragment=_parse_fragment(data["fragment"]),
            ingest=_parse_ingest(data["ingest"]),
            eval=_parse_eval(data["eval"]),
            leakage_probe=_parse_leakage_probe(data["leakage_probe"]),
            noise_floor=_parse_noise_floor(data["noise_floor"]),
            baseline=_parse_baseline(data["baseline"]),
        )


def _resolve(raw: str, repo_root: Path) -> Path:
    """Резолв пути из конфига относительно корня репозитория."""
    path = Path(raw).expanduser()
    return path if path.is_absolute() else repo_root / path


def _parse_paths(value: Any, *, repo_root: Path) -> PathsConfig:
    """Разобрать секцию ``paths``."""
    data = _mapping(value, "paths")
    _check_keys(
        data,
        "paths",
        required=(
            "data_dir",
            "raw_dir",
            "unified_dir",
            "fragments_dir",
            "splits_dir",
            "results_path",
        ),
    )
    data_dir = _resolve(_as_str(data["data_dir"], "paths.data_dir"), repo_root)
    return PathsConfig(
        repo_root=repo_root,
        data_dir=data_dir,
        raw_dir=_resolve(_as_str(data["raw_dir"], "paths.raw_dir"), repo_root),
        unified_dir=_resolve(_as_str(data["unified_dir"], "paths.unified_dir"), repo_root),
        fragments_dir=_resolve(_as_str(data["fragments_dir"], "paths.fragments_dir"), repo_root),
        splits_dir=_resolve(_as_str(data["splits_dir"], "paths.splits_dir"), repo_root),
        results_path=_resolve(_as_str(data["results_path"], "paths.results_path"), repo_root),
    )


def _parse_splits(value: Any) -> SplitsConfig:
    """Разобрать секцию ``splits``."""
    data = _mapping(value, "splits")
    _check_keys(
        data,
        "splits",
        required=("unit", "files", "train_datasets", "val_datasets", "test_datasets"),
    )
    unit = _as_str(data["unit"], "splits.unit")
    if unit not in SPLIT_UNITS:
        raise ConfigError(f"splits.unit: ожидалось одно из {SPLIT_UNITS}, получено {unit!r}")
    files = _mapping(data["files"], "splits.files")
    _check_keys(files, "splits.files", required=("train", "val", "test"))
    return SplitsConfig(
        unit=unit,  # type: ignore[arg-type]
        files={
            name: _as_str(files[name], f"splits.files.{name}") for name in ("train", "val", "test")
        },
        train_datasets=_as_str_tuple(data["train_datasets"], "splits.train_datasets"),
        val_datasets=_as_str_tuple(data["val_datasets"], "splits.val_datasets"),
        test_datasets=_as_str_tuple(data["test_datasets"], "splits.test_datasets"),
    )


def _parse_labels(value: Any) -> LabelsConfig:
    """Разобрать секцию ``labels`` и проверить инварианты §5."""
    data = _mapping(value, "labels")
    _check_keys(
        data,
        "labels",
        required=(
            "reference_only",
            "allow_interpolated",
            "max_soh_gap_cycles",
            "allow_extrapolation",
            "bol_source",
            "reference_cycle_rules",
        ),
    )
    reference_only = _as_bool(data["reference_only"], "labels.reference_only")
    allow_interpolated = _as_bool(data["allow_interpolated"], "labels.allow_interpolated")
    if allow_interpolated and reference_only:
        raise ConfigError(
            "labels: allow_interpolated=true требует reference_only=false "
            "(ТЗ §5.1: интерполяция возможна только поверх рабочих циклов)"
        )
    if _as_bool(data["allow_extrapolation"], "labels.allow_extrapolation"):
        raise ConfigError("labels.allow_extrapolation: экстраполяция запрещена ТЗ §5 п.5")
    max_gap = _as_int(data["max_soh_gap_cycles"], "labels.max_soh_gap_cycles")
    if max_gap < 1:
        raise ConfigError("labels.max_soh_gap_cycles: должен быть >= 1")
    rules = _mapping(data["reference_cycle_rules"], "labels.reference_cycle_rules")
    parsed_rules: dict[str, Mapping[str, Any]] = {}
    for dataset, rule in rules.items():
        parsed_rules[dataset] = dict(_mapping(rule, f"labels.reference_cycle_rules.{dataset}"))
    return LabelsConfig(
        reference_only=reference_only,
        allow_interpolated=allow_interpolated,
        max_soh_gap_cycles=max_gap,
        allow_extrapolation=False,
        bol_source=_as_str(data["bol_source"], "labels.bol_source"),
        reference_cycle_rules=parsed_rules,
    )


def _parse_step_labeling(value: Any) -> StepLabelingConfig:
    """Разобрать секцию ``step_labeling`` (ТЗ §4.4)."""
    data = _mapping(value, "step_labeling")
    _check_keys(
        data,
        "step_labeling",
        required=(
            "cc_current_tolerance_rel",
            "cv_voltage_tolerance_v",
            "min_step_points",
            "rest_current_tolerance_a",
            "current_sign_tolerance_a",
        ),
    )
    config = StepLabelingConfig(
        cc_current_tolerance_rel=_as_float(
            data["cc_current_tolerance_rel"], "step_labeling.cc_current_tolerance_rel"
        ),
        cv_voltage_tolerance_v=_as_float(
            data["cv_voltage_tolerance_v"], "step_labeling.cv_voltage_tolerance_v"
        ),
        min_step_points=_as_int(data["min_step_points"], "step_labeling.min_step_points"),
        rest_current_tolerance_a=_as_float(
            data["rest_current_tolerance_a"], "step_labeling.rest_current_tolerance_a"
        ),
        current_sign_tolerance_a=_as_float(
            data["current_sign_tolerance_a"], "step_labeling.current_sign_tolerance_a"
        ),
    )
    if config.cc_current_tolerance_rel <= 0 or config.cv_voltage_tolerance_v <= 0:
        raise ConfigError("step_labeling: допуски должны быть строго положительными")
    if config.min_step_points < 1:
        raise ConfigError("step_labeling.min_step_points: должен быть >= 1")
    return config


def _parse_fragment(value: Any) -> FragmentConfig:
    """Разобрать секцию ``fragment`` (ТЗ §7)."""
    data = _mapping(value, "fragment")
    _check_keys(
        data,
        "fragment",
        required=(
            "delta_v",
            "n_points",
            "phase",
            "v_normalization",
            "channels",
            "min_points_in_window",
            "seed",
            "windows_per_cycle",
            "delta_v_grid",
        ),
    )
    delta_v = _as_float(data["delta_v"], "fragment.delta_v")
    if delta_v <= 0:
        raise ConfigError("fragment.delta_v: должен быть > 0")
    n_points = _as_int(data["n_points"], "fragment.n_points")
    if n_points < 2:
        raise ConfigError("fragment.n_points: должен быть >= 2")
    min_points = _as_int(data["min_points_in_window"], "fragment.min_points_in_window")
    if min_points < 2:
        raise ConfigError("fragment.min_points_in_window: должен быть >= 2")
    phase = _as_str(data["phase"], "fragment.phase")
    if phase not in ("chg_cc", "chg_cv"):
        raise ConfigError(f"fragment.phase: ожидалась фаза заряда, получено {phase!r}")
    normalization = _as_str(data["v_normalization"], "fragment.v_normalization")
    if normalization not in V_NORMALIZATIONS:
        raise ConfigError(
            f"fragment.v_normalization: ожидалось одно из {V_NORMALIZATIONS}, "
            f"получено {normalization!r}"
        )
    channels = _as_str_tuple(data["channels"], "fragment.channels")
    unknown_channels = sorted(set(channels) - set(FRAGMENT_CHANNELS))
    if unknown_channels:
        raise ConfigError(
            f"fragment.channels: неизвестные каналы {unknown_channels}; "
            f"допустимые: {list(FRAGMENT_CHANNELS)}"
        )
    if "q_norm" not in channels:
        raise ConfigError("fragment.channels: канал q_norm обязателен (ТЗ §7)")
    grid = _as_float_tuple(data["delta_v_grid"], "fragment.delta_v_grid")
    if not grid or any(item <= 0 for item in grid):
        raise ConfigError("fragment.delta_v_grid: ожидался непустой список положительных чисел")
    return FragmentConfig(
        delta_v=delta_v,
        n_points=n_points,
        phase=phase,
        v_normalization=normalization,  # type: ignore[arg-type]
        channels=channels,
        min_points_in_window=min_points,
        seed=_as_int(data["seed"], "fragment.seed"),
        windows_per_cycle=_as_int(data["windows_per_cycle"], "fragment.windows_per_cycle"),
        delta_v_grid=grid,
    )


def _parse_ingest(value: Any) -> IngestConfig:
    """Разобрать секцию ``ingest`` (ТЗ §6)."""
    data = _mapping(value, "ingest")
    _check_keys(data, "ingest", required=("kit", "calce", "oxford", "sandia"))
    kit_data = _mapping(data["kit"], "ingest.kit")
    _check_keys(
        kit_data,
        "ingest.kit",
        required=("target_resolution_s", "keep_cc_full_resolution", "include_calendar_aging"),
    )
    resolution = _as_float(kit_data["target_resolution_s"], "ingest.kit.target_resolution_s")
    if resolution <= 0:
        raise ConfigError("ingest.kit.target_resolution_s: должен быть > 0")
    calce_data = _mapping(data["calce"], "ingest.calce")
    _check_keys(calce_data, "ingest.calce", required=("series",))
    return IngestConfig(
        kit=KitConfig(
            target_resolution_s=resolution,
            keep_cc_full_resolution=_as_bool(
                kit_data["keep_cc_full_resolution"], "ingest.kit.keep_cc_full_resolution"
            ),
            include_calendar_aging=_as_bool(
                kit_data["include_calendar_aging"], "ingest.kit.include_calendar_aging"
            ),
        ),
        calce=CalceConfig(series=_as_str_tuple(calce_data["series"], "ingest.calce.series")),
        oxford=dict(_mapping(data["oxford"], "ingest.oxford")),
        sandia=dict(_mapping(data["sandia"], "ingest.sandia")),
    )


def _parse_eval(value: Any) -> EvalConfig:
    """Разобрать секцию ``eval`` (ТЗ §10, §11)."""
    data = _mapping(value, "eval")
    _check_keys(
        data,
        "eval",
        required=(
            "main_test_chemistries",
            "hard_mode_chemistries",
            "calibration_coverage",
            "soh_deciles",
            "prediction_bounds",
        ),
    )
    bounds = _as_float_tuple(data["prediction_bounds"], "eval.prediction_bounds")
    if len(bounds) != 2 or bounds[0] >= bounds[1]:
        raise ConfigError("eval.prediction_bounds: ожидалась пара [lo, hi] с lo < hi")
    coverage = _as_float(data["calibration_coverage"], "eval.calibration_coverage")
    if not 0.0 < coverage < 1.0:
        raise ConfigError("eval.calibration_coverage: должно быть в (0, 1)")
    deciles = _as_int(data["soh_deciles"], "eval.soh_deciles")
    if deciles < 2:
        raise ConfigError("eval.soh_deciles: должно быть >= 2")
    return EvalConfig(
        main_test_chemistries=_as_str_tuple(
            data["main_test_chemistries"], "eval.main_test_chemistries"
        ),
        hard_mode_chemistries=_as_str_tuple(
            data["hard_mode_chemistries"], "eval.hard_mode_chemistries"
        ),
        calibration_coverage=coverage,
        soh_deciles=deciles,
        prediction_bounds=(bounds[0], bounds[1]),
    )


def _parse_leakage_probe(value: Any) -> LeakageProbeConfig:
    """Разобрать секцию ``leakage_probe`` (ТЗ §12.3)."""
    data = _mapping(value, "leakage_probe")
    _check_keys(data, "leakage_probe", required=("max_relative_improvement",))
    threshold = _as_float(
        data["max_relative_improvement"], "leakage_probe.max_relative_improvement"
    )
    if threshold < 0:
        raise ConfigError("leakage_probe.max_relative_improvement: должен быть >= 0")
    return LeakageProbeConfig(max_relative_improvement=threshold)


def _parse_noise_floor(value: Any) -> NoiseFloorConfig:
    """Разобрать секцию ``noise_floor`` (ТЗ §12.4)."""
    data = _mapping(value, "noise_floor")
    _check_keys(data, "noise_floor", required=("min_replicates", "replicate_keys"))
    replicates = _as_int(data["min_replicates"], "noise_floor.min_replicates")
    if replicates < 2:
        raise ConfigError("noise_floor.min_replicates: должен быть >= 2")
    return NoiseFloorConfig(
        min_replicates=replicates,
        replicate_keys=_as_str_tuple(data["replicate_keys"], "noise_floor.replicate_keys"),
    )


def _parse_baseline(value: Any) -> BaselineConfig:
    """Разобрать секцию ``baseline`` (ТЗ §12.5)."""
    data = _mapping(value, "baseline")
    _check_keys(data, "baseline", required=("model", "savgol", "elastic_net"))
    savgol = _mapping(data["savgol"], "baseline.savgol")
    _check_keys(savgol, "baseline.savgol", required=("window_length", "polyorder"))
    window_length = _as_int(savgol["window_length"], "baseline.savgol.window_length")
    polyorder = _as_int(savgol["polyorder"], "baseline.savgol.polyorder")
    if window_length % 2 == 0 or window_length < 3:
        raise ConfigError("baseline.savgol.window_length: должно быть нечётным и >= 3")
    if not 0 < polyorder < window_length:
        raise ConfigError("baseline.savgol.polyorder: должно быть в (0, window_length)")
    net = _mapping(data["elastic_net"], "baseline.elastic_net")
    _check_keys(net, "baseline.elastic_net", required=("alpha", "l1_ratio", "max_iter"))
    l1_ratio = _as_float(net["l1_ratio"], "baseline.elastic_net.l1_ratio")
    if not 0.0 <= l1_ratio <= 1.0:
        raise ConfigError("baseline.elastic_net.l1_ratio: должно быть в [0, 1]")
    return BaselineConfig(
        model=_as_str(data["model"], "baseline.model"),
        savgol_window_length=window_length,
        savgol_polyorder=polyorder,
        elastic_net_alpha=_as_float(net["alpha"], "baseline.elastic_net.alpha"),
        elastic_net_l1_ratio=l1_ratio,
        elastic_net_max_iter=_as_int(net["max_iter"], "baseline.elastic_net.max_iter"),
    )


# --------------------------------------------------------------------------
# Точка входа
# --------------------------------------------------------------------------


def default_config_path() -> Path:
    """Путь ``configs/benchmark.yaml`` в корне репозитория."""
    return Path(__file__).resolve().parents[2] / DEFAULT_CONFIG_NAME


def load_config(path: str | Path | None = None) -> BenchmarkConfig:
    """Загрузить и провалидировать конфиг бенчмарка.

    Parameters
    ----------
    path
        Путь к YAML. По умолчанию — ``configs/benchmark.yaml`` в корне репозитория.

    Raises
    ------
    ConfigError
        Если файл не найден, не является YAML-словарём или нарушает контракт.
    """
    resolved = Path(path).expanduser().resolve() if path is not None else default_config_path()
    return _load_config_cached(str(resolved))


@lru_cache(maxsize=8)
def _load_config_cached(resolved: str) -> BenchmarkConfig:
    """Кэшированная загрузка по абсолютному пути."""
    config_path = Path(resolved)
    if not config_path.is_file():
        raise ConfigError(f"конфиг не найден: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, Mapping):
        raise ConfigError(f"{config_path}: ожидался словарь верхнего уровня")
    repo_root = config_path.parent.parent
    return BenchmarkConfig.from_dict(raw, source_path=config_path, repo_root=repo_root)


def load_config_from_string(text: str, *, repo_root: Path | None = None) -> BenchmarkConfig:
    """Разобрать конфиг из строки (нужно тестам)."""
    raw = yaml.safe_load(text)
    if not isinstance(raw, Mapping):
        raise ConfigError("конфиг: ожидался словарь верхнего уровня")
    root = repo_root or Path.cwd()
    return BenchmarkConfig.from_dict(raw, source_path=root / DEFAULT_CONFIG_NAME, repo_root=root)
