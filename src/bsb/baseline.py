"""Бейзлайны и шумовой пол (ТЗ §12.1, §12.3, §12.4, §12.5).

Четыре публикуемых числа §12.1:

``constant``
    «пол» — предсказание среднего SOH обучающей выборки. Ниже него модель
    бесполезна.
``chem_constant``
    «константа по химии» — контроль для §12.3: предсказывает среднее SOH
    обучающей выборки **для своей химии**. Это лучший тривиальный предиктор для
    того, кто сумел опознать химию, но ничего не знает о старении. Сравнивать
    пробу надо именно с ним, а не только с глобальным «полом»: тест смешивает
    химии со средними SOH от 0.91 до 0.93, поэтому «пол» штрафуется за
    межхимический сдвиг, которого проба не устраняет, а обходит через уровень
    напряжения.
``probe``
    «проба на утечку» (§12.3): видит **только положение окна и метаданные
    ячейки**, но не саму кривую. Если проба заметно лучше **лучшей тривиальной
    константы**, генератор фрагментов протекает: допустимый диапазон ``v_start``
    сам коррелирует с SOH.
``elastic_net``
    собственно бейзлайн по признакам окна (§12.5).
``noise_floor``
    «потолок» (§12.4): разброс SOH между ячейками-репликами одного условия.
    Если бейзлайн ближе к потолку, чем к полу, — выигрыш закончился.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import polars as pl
from sklearn.linear_model import ElasticNet, LinearRegression

from bsb.config import BenchmarkConfig
from bsb.features import window_features

__all__ = [
    "MODEL_NAMES",
    "fit_predict",
    "noise_floor",
    "oracle_constant_mae",
    "probe_features",
]

#: Доступные модели (§12.5). ``probe`` намеренно не получает кривую.
#: ``chem_constant`` — не участник, а контроль §12.3: ему нужна химия теста,
#: которую организатор знает, а участник — нет.
MODEL_NAMES: tuple[str, ...] = ("constant", "chem_constant", "probe", "elastic_net")

#: Признаки пробы: только положение окна и метаданные, без формы кривой.
_PROBE_COLUMNS: tuple[str, ...] = ("f_v_start", "f_v_end", "f_v_width")


def probe_features(features: pl.DataFrame, metadata: pl.DataFrame) -> pl.DataFrame:
    """Собрать матрицу пробы: положение окна плюс метаданные ячейки (§12.3).

    Метаданные — ``nominal_capacity_ah`` и ``temperature_setpoint_c``: они
    известны организатору и не несут информации о текущем SOH, но позволяют
    пробе выучить смещения между датасетами.
    """
    joined = features.join(metadata, on="fragment_id", how="left")
    columns = [*_PROBE_COLUMNS, "nominal_capacity_ah", "temperature_setpoint_c"]
    return joined.select(["fragment_id", *columns])


def _design(
    fragments: pl.DataFrame,
    config: BenchmarkConfig,
    metadata: pl.DataFrame,
    *,
    model: str,
) -> tuple[np.ndarray, np.ndarray, list[str], pl.DataFrame]:
    """Матрица признаков и цель для выбранной модели."""
    features = window_features(fragments, config)
    if model == "probe":
        features = probe_features(features, metadata)
    columns = [column for column in features.columns if column not in {"fragment_id", "soh"}]
    matrix = features.select(columns).to_numpy().astype(float)
    target_frame = fragments.select("fragment_id", "soh")
    features = features.select("fragment_id").join(target_frame, on="fragment_id", how="left")
    target = features["soh"].to_numpy().astype(float)
    return matrix, target, columns, features.select("fragment_id")


def oracle_constant_mae(truth: pl.DataFrame, chemistries: Sequence[str]) -> float:
    """MAE константы, подобранной на самом тест-домене (диагностика §12.3).

    Это сильнейший предиктор, который не смотрит на форму кривой: «среднее SOH
    этих условий». Участнику он доступен — каналы температуры и тока выдают
    домен, — поэтому при кросс-датасетном тесте честным нулём для пробы
    является именно он, а не среднее обучения: среднее обучения смещено
    (Barkholtz почти свежие, SNL постаревшие) и штрафует пробу за то, что она
    этот сдвиг компенсирует.

    Returns
    -------
    float
        MAE в процентных пунктах; ``nan``, если химий в эталоне нет.
    """
    selected = truth.filter(pl.col("chemistry").is_in(list(chemistries)))
    if selected.height == 0:
        return float("nan")
    mean = float(selected["soh"].mean())
    return float((selected["soh"] - mean).abs().mean() * 100)


def _chem_constant_predict(
    train: pl.DataFrame,
    test: pl.DataFrame,
    metadata: pl.DataFrame,
) -> np.ndarray:
    """Предсказать среднее SOH обучающей выборки по химии тестового фрагмента.

    Химия берётся из ``metadata`` (её знает организатор). Для химии, не
    встречавшейся в обучении, используется глобальное среднее — тогда поведение
    совпадает с ``constant``.
    """
    if "chemistry" not in metadata.columns:
        raise ValueError("для модели 'chem_constant' в metadata нужна колонка 'chemistry'")
    chemistry = metadata.select("fragment_id", "chemistry")
    means = (
        train.select("fragment_id", "soh")
        .join(chemistry, on="fragment_id", how="left")
        .group_by("chemistry")
        .agg(pl.col("soh").mean().alias("chem_mean"))
    )
    fallback = float(train["soh"].mean())
    joined = (
        test.select("fragment_id")
        .join(chemistry, on="fragment_id", how="left")
        .join(means, on="chemistry", how="left")
    )
    return joined["chem_mean"].fill_null(fallback).to_numpy().astype(float)


def fit_predict(
    train: pl.DataFrame,
    test: pl.DataFrame,
    *,
    model: str,
    config: BenchmarkConfig,
    metadata: pl.DataFrame,
) -> pl.DataFrame:
    """Обучить модель на ``train`` и предсказать ``test``.

    Parameters
    ----------
    train, test
        Размеченные фрагменты сплитов (``train.parquet`` / ``val.parquet``).
    model
        ``constant``, ``probe`` или ``elastic_net`` (§12.5).
    metadata
        Соответствие ``fragment_id`` → метаданные ячейки; нужно пробе.

    Returns
    -------
    polars.DataFrame
        ``fragment_id`` и ``soh_pred`` (для ``elastic_net`` — ещё и признаки
        важности через коэффициенты не возвращаем: их смотрят отдельно).
    """
    if model not in MODEL_NAMES:
        raise ValueError(f"неизвестная модель {model!r}; доступны {MODEL_NAMES}")

    if model == "chem_constant":
        prediction = _chem_constant_predict(train, test, metadata)
        low, high = config.eval.prediction_bounds
        prediction = np.clip(prediction, low, high)
        ids = test.select("fragment_id")
        return pl.DataFrame({"fragment_id": ids["fragment_id"], "soh_pred": prediction})

    train_matrix, train_target, _, _ = _design(train, config, metadata, model=model)
    test_matrix, _, _, test_ids = _design(test, config, metadata, model=model)
    if train_matrix.shape[0] == 0 or test_matrix.shape[0] == 0:
        raise ValueError("пустая обучающая или тестовая выборка")

    if model == "constant":
        prediction = np.full(test_matrix.shape[0], float(np.mean(train_target)))
    else:
        estimator = (
            LinearRegression()
            if model == "probe"
            else ElasticNet(
                alpha=config.baseline.elastic_net_alpha,
                l1_ratio=config.baseline.elastic_net_l1_ratio,
                max_iter=config.baseline.elastic_net_max_iter,
                random_state=config.seed,
            )
        )
        # Масштаб признаков разный (напряжения, ёмкости, производные), поэтому
        # центрируем и нормируем по обучающей выборке — иначе ElasticNet
        # штрафует признаки неравномерно.
        centre = train_matrix.mean(axis=0)
        scale = train_matrix.std(axis=0)
        scale[scale < 1e-12] = 1.0
        estimator.fit((train_matrix - centre) / scale, train_target)
        prediction = estimator.predict((test_matrix - centre) / scale)

    low, high = config.eval.prediction_bounds
    prediction = np.clip(prediction, low, high)
    return pl.DataFrame({"fragment_id": test_ids["fragment_id"], "soh_pred": prediction})


def noise_floor(
    cycles: pl.DataFrame,
    cells: pl.DataFrame,
    config: BenchmarkConfig,
    *,
    datasets: Sequence[str] | None = None,
    points: int = 50,
) -> dict[str, object]:
    """Оценить шумовой пол по ячейкам-репликам (ТЗ §12.4).

    Условие — набор ``noise_floor.replicate_keys`` (по умолчанию химия и
    уставка температуры). Внутри условия берутся группы ячеек; для каждой
    кривая SOH интерполируется на общую относительную шкалу жизни, после чего
    считается разброс между ячейками.

    Потолок считается **по датасетам теста** (``datasets``): условие имеет смысл
    там, где ячейки реально состарились. Общая медиана по всем наборам
    обманчива — почти свежие ячейки архива Sandia дают разброс ~0.1 п.п. и
    вытягивают её вниз, тогда как на тестовых условиях разброс 5-9 п.п.

    Returns
    -------
    dict
        ``noise_floor_pp`` — медиана разброса по условиям теста в процентных
        пунктах, ``conditions``, ``replicates``, ``per_condition`` — разбивка,
        ``by_dataset`` — медианы по наборам (для сравнения).
    """
    keys = list(config.noise_floor.replicate_keys)
    dataset_key = "dataset" if "dataset" in cells.columns else None
    group_keys = [*keys]
    if dataset_key and dataset_key not in group_keys:
        # Условия из разных датасетов несопоставимы по определению SOH.
        group_keys = [dataset_key, *group_keys]

    joined = cycles.join(cells.select(["cell_id", *group_keys]), on="cell_id", how="left")
    selected = list(datasets) if datasets else list(config.splits.test_datasets)
    per_condition: list[dict[str, object]] = []
    spreads: list[float] = []
    for condition, part in joined.group_by(group_keys, maintain_order=True):
        labels = condition if isinstance(condition, tuple) else (condition,)
        by_cell = part.group_by("cell_id").len()
        if by_cell.height < config.noise_floor.min_replicates:
            continue
        grid = np.linspace(0.0, 1.0, points)
        curves = []
        for _cell_id, cell_cycles in part.group_by("cell_id", maintain_order=True):
            ordered = cell_cycles.sort("cycle_index")
            soh = ordered["soh"].to_numpy()
            if soh.size < config.noise_floor.min_replicates:
                continue
            life = np.linspace(0.0, 1.0, soh.size)
            curves.append(np.interp(grid, life, soh))
        if len(curves) < config.noise_floor.min_replicates:
            continue
        stack = np.vstack(curves)
        spread = float(np.mean(stack.std(axis=0)))
        spreads.append(spread)
        per_condition.append(
            {
                **dict(zip(group_keys, labels, strict=True)),
                "cells": len(curves),
                "spread_pp": round(spread * 100.0, 4),
            }
        )

    by_dataset: dict[str, float] = {}
    for dataset in sorted({str(item["dataset"]) for item in per_condition}):
        values = [item["spread_pp"] for item in per_condition if str(item["dataset"]) == dataset]
        if values:
            by_dataset[dataset] = round(float(np.median(values)), 4)
    test_spreads = [item["spread_pp"] for item in per_condition if str(item["dataset"]) in selected]
    if not test_spreads:
        return {
            "noise_floor_pp": None,
            "conditions": 0,
            "replicates": 0,
            "per_condition": per_condition,
            "by_dataset": by_dataset,
            "note": f"в датасетах теста {selected} нет условий с нужным числом реплик",
        }
    return {
        "noise_floor_pp": round(float(np.median(test_spreads)), 4),
        "datasets": selected,
        "conditions": len(test_spreads),
        "replicates": sum(
            int(item["cells"]) for item in per_condition if str(item["dataset"]) in selected
        ),
        "per_condition": sorted(per_condition, key=lambda item: item["spread_pp"], reverse=True),
        "by_dataset": by_dataset,
    }
