"""Тесты бейзлайнов и контроля §12.3 (:mod:`bsb.baseline`)."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import polars as pl
import pytest

from bsb.baseline import MODEL_NAMES, fit_predict, oracle_constant_mae


def _fragments(rows: list[tuple[str, str, float]]) -> pl.DataFrame:
    """Минимальный кадр фрагментов: id, ячейка и целевой SOH.

    ``chem_constant`` не смотрит на кривую, поэтому трёх колонок достаточно:
    остальные признаки требуются только ``probe`` и ``elastic_net``.
    """
    return pl.DataFrame(
        {
            "fragment_id": [row[0] for row in rows],
            "cell_id": [row[1] for row in rows],
            "soh": [row[2] for row in rows],
        }
    )


def _metadata(pairs: list[tuple[str, str]]) -> pl.DataFrame:
    return pl.DataFrame({"fragment_id": [p[0] for p in pairs], "chemistry": [p[1] for p in pairs]})


def test_model_names_expose_leakage_controls() -> None:
    assert set(MODEL_NAMES) == {"constant", "chem_constant", "probe", "elastic_net"}


def test_chem_constant_predicts_training_mean_of_own_chemistry(
    make_config: Callable[[dict[str, Any] | None], Any],
) -> None:
    """Контроль §12.3 отвечает средним SOH своей химии, а не общим средним."""
    config = make_config(None)
    train = _fragments([("a1", "c1", 0.8), ("a2", "c1", 1.0), ("b1", "c2", 0.5)])
    test = _fragments([("t1", "c3", 0.0), ("t2", "c4", 0.0)])
    metadata = _metadata(
        [("a1", "LFP"), ("a2", "LFP"), ("b1", "NMC"), ("t1", "LFP"), ("t2", "NMC")]
    )

    prediction = fit_predict(train, test, model="chem_constant", config=config, metadata=metadata)
    values = prediction.sort("fragment_id")["soh_pred"].to_list()

    assert values == pytest.approx([0.9, 0.5])


def test_chem_constant_falls_back_to_global_mean_for_unseen_chemistry(
    make_config: Callable[[dict[str, Any] | None], Any],
) -> None:
    """Незнакомая химия не должна ронять счёт — берём глобальное среднее."""
    config = make_config(None)
    train = _fragments([("a1", "c1", 0.4), ("a2", "c1", 0.6)])
    metadata = _metadata([("a1", "LFP"), ("a2", "LFP"), ("t1", "NCA")])

    prediction = fit_predict(
        train,
        _fragments([("t1", "c9", 0.0)]),
        model="chem_constant",
        config=config,
        metadata=metadata,
    )

    assert prediction["soh_pred"].to_list() == pytest.approx([0.5])


def test_chem_constant_requires_chemistry_in_metadata(
    make_config: Callable[[dict[str, Any] | None], Any],
) -> None:
    """Без химии контроль неопределён — это ошибка вызова, а не молчаливый пол."""
    config = make_config(None)
    train = _fragments([("a1", "c1", 0.9)])
    metadata = _metadata([("a1", "LFP"), ("t1", "LFP")]).drop("chemistry")

    with pytest.raises(ValueError, match="chemistry"):
        fit_predict(
            train,
            _fragments([("t1", "c1", 0.0)]),
            model="chem_constant",
            config=config,
            metadata=metadata,
        )


def test_oracle_constant_mae_is_mean_absolute_deviation_in_points(
    make_config: Callable[[dict[str, Any] | None], Any],
) -> None:
    """Ноль §12.3 — константа тест-домена, MAE считается в процентных пунктах."""
    truth = pl.DataFrame(
        {
            "fragment_id": ["t1", "t2", "t3", "t4"],
            "chemistry": ["LFP", "LFP", "NCA", "NCA"],
            "soh": [0.8, 1.0, 0.6, 0.9],
        }
    )

    # LFP: среднее 0.9, отклонения 0.1 + 0.1 → 10 п.п.
    assert oracle_constant_mae(truth, ["LFP"]) == pytest.approx(10.0)
    # NCA: среднее 0.75, отклонения 0.15 + 0.15 → 15 п.п.
    assert oracle_constant_mae(truth, ["NCA"]) == pytest.approx(15.0)


def test_oracle_constant_mae_is_nan_without_matching_chemistry() -> None:
    """Пустая номинация не должна давать ноль — это отсутствие данных."""
    truth = pl.DataFrame({"fragment_id": ["t1"], "chemistry": ["LFP"], "soh": [0.9]})

    assert math.isnan(oracle_constant_mae(truth, ["NMC"]))
