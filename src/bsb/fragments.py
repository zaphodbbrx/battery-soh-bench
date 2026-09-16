"""Генератор фрагментов (ТЗ §7) — единственная точка порождения входов модели.

Алгоритм §7 для одного цикла:

1. взять CC-фазу заряда (``step_type = fragment.phase``);
2. выбрать начало окна ``v_start`` равномерно из допустимого диапазона так,
   чтобы ``[v_start, v_start + delta_v]`` целиком лежало внутри фазы;
3. интерполировать ``Q(V)`` на равномерную сетку из ``n_points``;
4. нормировать ``Q`` на ``bol_capacity_ah`` этой ячейки;
5. приложить каналы тока и температуры на той же сетке.

Режимы §7
---------
``fixed`` (:func:`generate_fragments`)
    детерминированная генерация по ``fragment.seed``. ГПСЧ заводится от
    ``(seed, cell_id, cycle_index, номер окна)``, а не от глобального
    состояния, поэтому порядок обхода ячеек на результат не влияет.
    Используется для val и test, результат коммитится артефактом.

``sampling`` (:func:`iter_sampled_windows`)
    случайные окна на лету, для аугментации на обучении; наружу отдаётся через
    :func:`bsb.dataset.sample_fragments`. Результат намеренно не воспроизводим,
    поэтому в артефакт val/test такие фрагменты попадать не должны.

``v_normalization`` (§7, §9)
----------------------------
``per_cell_range``
    допустимый диапазон ``v_start`` берётся у самой ячейки (по всем её
    CC-фазам), поэтому окна существуют для любой химии, включая LFP;
``absolute``
    диапазон берётся из ``fragment.absolute_bounds``; на LFP окна тогда
    физически не существуют и фрагментов не будет — это предсказание §9, а не
    ошибка.

Предупреждение §12.3: при ``per_cell_range`` допустимый диапазон ``v_start``
сам коррелирует с SOH, поэтому позицию окна нельзя подавать модели как признак
без проверки пробой на утечку (§12.3, ``bsb baseline --model probe``).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import polars as pl

from bsb.config import BenchmarkConfig
from bsb.schema import FRAGMENTS_SCHEMA, FRAGMENTS_UNLABELED_SCHEMA, CellBundle, conform

__all__ = [
    "FragmentWindow",
    "fragment_id",
    "generate_fragments",
    "iter_fragment_windows",
    "iter_sampled_windows",
    "strip_labels",
]


@dataclass(frozen=True, slots=True)
class FragmentWindow:
    """Одно окно: границы, сетка напряжений и каналы на ней."""

    cycle_index: int
    v_start: float
    v_end: float
    q_norm: np.ndarray
    current: np.ndarray
    temperature: np.ndarray


@dataclass(frozen=True, slots=True)
class _PhasePlan:
    """Подготовленные данные ячейки: фазы заряда, границы окон и ``bol``."""

    phases: dict[int, pl.DataFrame]
    bounds: tuple[float, float] | None
    bol: float


def _digest(*parts: object, size: int = 8) -> bytes:
    """Стабильный (не зависящий от запуска) хеш набора значений."""
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return hashlib.blake2b(payload, digest_size=size).digest()


def fragment_id(cell_id: str, cycle_index: int, v_start: float, seed: int) -> str:
    """Детерминированный идентификатор фрагмента (ТЗ §7).

    Хеш от ``(cell_id, cycle_index, v_start, seed)``, а не счётчик: он
    воспроизводится при любой повторной генерации и не зависит от порядка
    обхода ячеек.
    """
    return _digest(cell_id, cycle_index, f"{v_start:.6f}", seed, size=8).hex()


def _rng(seed: int, cell_id: str, cycle_index: int, window: int) -> np.random.Generator:
    """ГПСЧ, привязанный к конкретному окну (детерминизм §7)."""
    return np.random.default_rng(int.from_bytes(_digest(seed, cell_id, cycle_index, window), "big"))


def _cc_phase(timeseries: pl.DataFrame, cycle_index: int, phase: str) -> pl.DataFrame:
    """CC-фаза заряда указанного цикла, отсортированная по напряжению.

    Дубликаты напряжения ломают ``np.interp``, поэтому для каждого уровня
    остаётся первое вхождение.
    """
    phase_rows = timeseries.filter(
        (pl.col("cycle_index") == cycle_index) & (pl.col("step_type") == pl.lit(phase))
    )
    if phase_rows.height == 0:
        return phase_rows
    phase_rows = phase_rows.sort("voltage_v").with_columns(
        pl.col("voltage_v").is_first_distinct().alias("_first")
    )
    return phase_rows.filter(pl.col("_first")).drop("_first")


def _plan(bundle: CellBundle, config: BenchmarkConfig) -> _PhasePlan:
    """Собрать CC-фазы размеченных циклов и допустимые границы окон."""
    fragment = config.fragment
    labelled = bundle.cycles
    phases: dict[int, pl.DataFrame] = {}
    for row in labelled["cycle_index"].to_list():
        phase = _cc_phase(bundle.timeseries, int(row), fragment.phase)
        if phase.height > 1:
            phases[int(row)] = phase
    if not phases:
        return _PhasePlan(phases={}, bounds=None, bol=float(bundle.cell["bol_capacity_ah"]))

    if fragment.v_normalization == "per_cell_range":
        v_min = min(float(phase["voltage_v"].min()) for phase in phases.values())
        v_max = max(float(phase["voltage_v"].max()) for phase in phases.values())
    else:
        v_min, v_max = fragment.absolute_bounds
    upper = v_max - fragment.delta_v
    bounds = None if upper < v_min else (float(v_min), float(upper))
    return _PhasePlan(phases=phases, bounds=bounds, bol=float(bundle.cell["bol_capacity_ah"]))


def _make_window(
    phase: pl.DataFrame,
    *,
    cycle_index: int,
    bounds: tuple[float, float],
    rng: np.random.Generator,
    config: BenchmarkConfig,
    bol: float,
) -> FragmentWindow | None:
    """Одно окно: выбор ``v_start``, проверка плотности, ресемплинг каналов."""
    fragment = config.fragment
    low, high = bounds
    v_start = float(rng.uniform(low, high)) if high > low else float(low)
    v_end = v_start + fragment.delta_v
    inside = phase.filter((pl.col("voltage_v") >= v_start) & (pl.col("voltage_v") <= v_end))
    if inside.height < fragment.min_points_in_window:
        return None
    grid = np.linspace(v_start, v_end, fragment.n_points)
    voltage = phase["voltage_v"].to_numpy()
    charge = np.interp(grid, voltage, phase["charge_ah"].to_numpy())
    current = np.interp(grid, voltage, phase["current_a"].to_numpy())
    temperature = np.interp(grid, voltage, phase["temperature_c"].to_numpy())
    return FragmentWindow(
        cycle_index=cycle_index,
        v_start=v_start,
        v_end=v_end,
        q_norm=charge / bol,
        current=current,
        temperature=temperature,
    )


def iter_fragment_windows(
    bundle: CellBundle,
    config: BenchmarkConfig,
    *,
    cycle_index: int | None = None,
) -> list[FragmentWindow]:
    """Детерминированные окна ячейки: ``fragment.windows_per_cycle`` на цикл.

    Parameters
    ----------
    cycle_index
        Ограничить генерацию одним циклом (нужно тестам и отладке).
    """
    fragment = config.fragment
    plan = _plan(bundle, config)
    if plan.bounds is None:
        return []
    selected = sorted(plan.phases) if cycle_index is None else [cycle_index]
    windows: list[FragmentWindow] = []
    for index in selected:
        phase = plan.phases.get(index)
        if phase is None:
            continue
        for slot in range(fragment.windows_per_cycle):
            rng = _rng(fragment.seed, bundle.cell_id, index, slot)
            window = _make_window(
                phase,
                cycle_index=index,
                bounds=plan.bounds,
                rng=rng,
                config=config,
                bol=plan.bol,
            )
            if window is not None:
                windows.append(window)
    return windows


def iter_sampled_windows(
    bundle: CellBundle,
    config: BenchmarkConfig,
    rng: np.random.Generator,
    *,
    windows: int = 1,
    cycle_index: int | None = None,
) -> list[FragmentWindow]:
    """Случайные окна для аугментации (режим ``sampling`` ТЗ §7).

    Каждое окно рисуется из переданного ``rng``, поэтому выборка не
    воспроизводима и не должна попадать в артефакты val/test.
    """
    plan = _plan(bundle, config)
    if plan.bounds is None:
        return []
    selected = sorted(plan.phases) if cycle_index is None else [cycle_index]
    produced: list[FragmentWindow] = []
    for _ in range(windows):
        for index in selected:
            phase = plan.phases.get(index)
            if phase is None:
                continue
            window = _make_window(
                phase,
                cycle_index=index,
                bounds=plan.bounds,
                rng=rng,
                config=config,
                bol=plan.bol,
            )
            if window is not None:
                produced.append(window)
    return produced


def _frame(
    rows: list[dict[str, object]],
    *,
    labeled: bool,
) -> pl.DataFrame:
    """Собрать кадр фрагментов нужной схемы (с метками или без)."""
    schema = FRAGMENTS_SCHEMA if labeled else FRAGMENTS_UNLABELED_SCHEMA
    if not rows:
        return pl.DataFrame(schema=dict(schema))
    frame = pl.DataFrame(rows, schema=dict(FRAGMENTS_SCHEMA))
    if not labeled:
        frame = frame.drop(["cell_id", "cycle_index", "soh"])
    return conform(
        frame, FRAGMENTS_SCHEMA if labeled else FRAGMENTS_UNLABELED_SCHEMA, name="fragments"
    )


def _rows_for(
    bundle: CellBundle,
    windows: list[FragmentWindow],
    config: BenchmarkConfig,
) -> list[dict[str, object]]:
    """Превратить окна в строки фрагментов с метками."""
    seed = config.fragment.seed
    soh_by_cycle = dict(
        zip(bundle.cycles["cycle_index"].to_list(), bundle.cycles["soh"].to_list(), strict=True)
    )
    return [
        {
            "fragment_id": fragment_id(bundle.cell_id, window.cycle_index, window.v_start, seed),
            "cell_id": bundle.cell_id,
            "cycle_index": window.cycle_index,
            "v_start": window.v_start,
            "v_end": window.v_end,
            "q_norm": window.q_norm.tolist(),
            "current": window.current.tolist(),
            "temperature": window.temperature.tolist(),
            "soh": float(soh_by_cycle[window.cycle_index]),
        }
        for window in windows
    ]


def generate_fragments(
    bundle: CellBundle,
    config: BenchmarkConfig,
    *,
    labeled: bool = True,
) -> pl.DataFrame:
    """Сгенерировать фрагменты одной ячейки в режиме ``fixed`` (ТЗ §7)."""
    windows = iter_fragment_windows(bundle, config)
    return _frame(_rows_for(bundle, windows, config), labeled=labeled)


def strip_labels(fragments: pl.DataFrame) -> pl.DataFrame:
    """Убрать ``cell_id``, ``cycle_index`` и ``soh`` для тестового файла (ТЗ §7)."""
    return conform(
        fragments.drop(["cell_id", "cycle_index", "soh"]),
        FRAGMENTS_UNLABELED_SCHEMA,
        name="fragments (unlabeled)",
    )
