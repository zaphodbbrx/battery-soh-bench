"""Обучение модели SOH по фрагменту зарядной кривой и подготовка сабмита.

Запуск из корня репозитория:

    uv run python submissions/chedum-team/train.py

Шаги:
1. признаки окна: форма dQ/dV, нормированная внутри окна, и относительный
   заряд (по сравнению с «почти новой» ячейкой при том же напряжении);
2. проверка запоминания: ошибка на незнакомых ячейках (GroupKFold) против
   ошибки на своих обучающих данных;
3. проверка переноса на отложенные домены (Sandia NCA, Oxford);
4. отчёт по val: прогноз, истинный SOH, отклонение;
5. финальная модель на train + val -> прогноз для test_unlabeled.
"""

from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path

import numpy as np
import polars as pl
from scipy.signal import savgol_filter
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "fragments"
OUT = Path(__file__).resolve().parent
RESULTS = OUT / "results"
SHAPE_POINTS = 16
SOH_BOUNDS = (0.45, 1.05)


# ---------------------------------------------------------------- признаки
def fragment_features(
    q: np.ndarray, cur: np.ndarray, temp: np.ndarray, v_start: float, v_end: float
) -> dict[str, float]:
    """Признаки одного окна."""
    grid = np.linspace(v_start, v_end, q.size)
    smooth = savgol_filter(q, 11, 3) if q.size > 11 else q
    dqdv = np.gradient(smooth, grid)
    dq = float(q[-1] - q[0])
    mean_dqdv = float(np.mean(dqdv))
    scale = mean_dqdv if abs(mean_dqdv) > 1e-9 else 1e-9
    shape = dqdv / scale  # форма производной, нормированная внутри окна
    shape_binned = np.array([b.mean() for b in np.array_split(shape, SHAPE_POINTS)])
    qn = (q - q[0]) / (dq if abs(dq) > 1e-12 else 1e-12)  # q, нормированное в [0, 1]
    current = float(np.mean(cur))

    feats: dict[str, float] = {
        "v_start": v_start,
        "v_mid": (v_start + v_end) / 2,
        "dq_window": dq,  # сколько заряда в окне 0.2 В
        "log_dq_window": float(np.log(max(dq, 1e-4))),
        "q_start": float(q[0]),
        "q_end": float(q[-1]),
        "dqdv_mean": mean_dqdv,
        "dqdv_peak_pos": float(np.argmax(dqdv) / (q.size - 1)),
        "dqdv_peak_rel": float(np.max(shape)),
        "dqdv_min_rel": float(np.min(shape)),
        "dqdv_std_rel": float(np.std(shape)),
        "dqdv_slope_rel": float(np.polyfit(np.linspace(0, 1, q.size), shape, 1)[0]),
        "q_half_pos": float(np.searchsorted(qn, 0.5) / (q.size - 1)),
        "q_curv": float(np.mean(qn) - 0.5),
        "current": current,
        "log_current": float(np.log(max(abs(current), 1e-3))),
        "temperature": float(np.mean(temp)),
    }
    for i, value in enumerate(shape_binned):
        feats[f"shape_{i:02d}"] = float(value)
    return feats


def build_features(frame: pl.DataFrame) -> pl.DataFrame:
    """Посчитать признаки окна для всех фрагментов набора."""
    rows = []
    for r in frame.iter_rows(named=True):
        f = fragment_features(
            np.asarray(r["q_norm"], float),
            np.asarray(r["current"], float),
            np.asarray(r["temperature"], float),
            float(r["v_start"]),
            float(r["v_end"]),
        )
        rows.append(f)
    return pl.DataFrame(rows)


def protocol_group(feats: pl.DataFrame) -> np.ndarray:
    """Группа по уровню тока заряда (лог-шкала): по сути, протокол / тип ячейки."""
    current = np.abs(feats["current"].to_numpy()) + 1e-3
    return np.round(np.log(current) / 0.15).astype(int)


def relative_features(
    feats: pl.DataFrame,
    quantile: float = 95,
    half_width: float = 0.05,
) -> pl.DataFrame:
    """Заряд относительно «почти новой» ячейки при том же напряжении.

    Для каждого фрагмента берутся фрагменты той же группы тока с близким
    положением окна (±50 мВ), и их 95-й процентиль считается эталоном
    неизношенной ячейки. Признак = значение / эталон. Метки SOH не нужны,
    поэтому признак одинаково считается и на train, и на test, и не зависит
    от абсолютного масштаба кривой конкретной химии.
    Считается отдельно внутри каждого набора (train, val, test).
    """
    groups = protocol_group(feats)
    v_mid = feats["v_mid"].to_numpy()
    out: dict[str, np.ndarray] = {}
    for name in ("q_end", "q_start", "dq_window"):
        values = feats[name].to_numpy()
        rel = np.empty_like(values)
        for g in np.unique(groups):
            idx = np.where(groups == g)[0]
            for i in idx:
                near = idx[np.abs(v_mid[idx] - v_mid[i]) < half_width]
                rel[i] = values[i] / max(np.percentile(values[near], quantile), 1e-6)
        out[f"rel_{name}"] = np.clip(rel, 0.0, 1.5)
    return pl.DataFrame(out)


SHAPE_COLS = [f"shape_{i:02d}" for i in range(SHAPE_POINTS)] + [
    "dqdv_peak_pos",
    "dqdv_peak_rel",
    "dqdv_min_rel",
    "dqdv_std_rel",
    "dqdv_slope_rel",
    "q_half_pos",
    "q_curv",
]
REL_COLS = ["rel_q_end", "rel_q_start", "rel_dq_window"]
#: Набор признаков, устойчивый к смене домена: относительный заряд, форма кривой,
#: ток и температура. Абсолютные q и положение окна исключены — они «узнают» датасет.
ROBUST_COLS = REL_COLS + SHAPE_COLS + ["log_current", "temperature"]


def full_features(frame: pl.DataFrame) -> pl.DataFrame:
    """Базовые признаки окна плюс относительные (внутри набора)."""
    feats = build_features(frame)
    return feats.hstack(relative_features(feats))


def soh_correlations(
    feats: pl.DataFrame, soh: np.ndarray, domains: np.ndarray
) -> dict[str, dict[str, float]]:
    """Корреляция Пирсона «признак — SOH» внутри каждого домена.

    Сравниваются сырые значения заряда и их относительные версии: это проверка
    того, что относительный признак действительно несёт информацию об износе.
    """
    out: dict[str, dict[str, float]] = {}
    names = ["q_end", "q_start", "dq_window", *REL_COLS]
    for domain in sorted(set(domains)):
        mask = domains == domain
        if soh[mask].std() == 0:
            continue
        out[domain] = {
            name: round(float(np.corrcoef(feats[name].to_numpy()[mask], soh[mask])[0, 1]), 3)
            for name in names
        }
    return out


# ---------------------------------------------------------------- модели
def _extra_trees(leaf: int) -> ExtraTreesRegressor:
    return ExtraTreesRegressor(
        n_estimators=400,
        min_samples_leaf=leaf,
        max_features=0.6,
        n_jobs=-1,
        random_state=SEED,
    )


def candidates() -> dict[str, tuple[str, object]]:
    """Кандидаты: имя -> (набор признаков, модель).

    ``min_samples_leaf`` у деревьев — главный рычаг против запоминания: лист
    не может описывать меньше этого числа фрагментов, то есть модель не
    подстраивается под отдельные циклы конкретной ячейки.
    """
    return {
        "mean (пол)": ("базовые", DummyRegressor(strategy="mean")),
        "linear_regression": ("базовые", make_pipeline(StandardScaler(), LinearRegression())),
        "ridge": ("базовые", make_pipeline(StandardScaler(), Ridge(alpha=10.0))),
        "elastic_net": (
            "базовые",
            make_pipeline(StandardScaler(), ElasticNet(alpha=0.002, l1_ratio=0.5, max_iter=20000)),
        ),
        "extra_trees leaf=5": ("устойчивые", _extra_trees(5)),
        "extra_trees leaf=20": ("устойчивые", _extra_trees(20)),
        "extra_trees leaf=50": ("устойчивые", _extra_trees(50)),
        "hist_gbr depth=3": (
            "устойчивые",
            HistGradientBoostingRegressor(
                max_depth=3,
                min_samples_leaf=50,
                learning_rate=0.05,
                max_iter=300,
                l2_regularization=1.0,
                random_state=SEED,
            ),
        ),
    }


def cell_weights(cells: np.ndarray) -> np.ndarray:
    """Вес фрагмента ~ 1/sqrt(число фрагментов ячейки): CALCE не задавливает остальных."""
    _, inv, counts = np.unique(cells, return_inverse=True, return_counts=True)
    w = 1.0 / np.sqrt(counts[inv])
    return w / w.mean()


def fit(model: object, x: np.ndarray, y: np.ndarray, w: np.ndarray) -> object:
    """Обучить копию модели с весами фрагментов."""
    m = clone(model)
    last = m.steps[-1][0] + "__sample_weight" if hasattr(m, "steps") else "sample_weight"
    m.fit(x, y, **{last: w})
    return m


def predict(model: object, x: np.ndarray) -> np.ndarray:
    """Прогноз, обрезанный до допустимого диапазона SOH."""
    return np.clip(model.predict(x), *SOH_BOUNDS)


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """MAE, P95 и смещение в процентных пунктах SOH."""
    err = p - y
    return {
        "n": int(y.size),
        "mae_pp": round(float(np.mean(np.abs(err))) * 100, 3),
        "p95_pp": round(float(np.percentile(np.abs(err), 95)) * 100, 3),
        "bias_pp": round(float(np.mean(err)) * 100, 3),
    }


def dataset_of(cell: str) -> str:
    """Домен ячейки по её идентификатору."""
    if cell.startswith("sandia_rd_"):
        return "sandia_" + cell.split("_")[2]  # sandia_LCO / NCA / NMC / LFP
    return cell.split("_")[0]


# ---------------------------------------------------------------- main
def lodo_eval(
    model: object, pool: pl.DataFrame, cols: list[str], domains: np.ndarray, holdouts: list[str]
) -> dict[str, dict[str, float]]:
    """Перенос на незнакомый домен: учим без него, проверяем на нём.

    Относительные признаки пересчитываются отдельно для обучающей части и для
    отложенного домена — ровно так, как это будет для теста.
    """
    y = pool["soh"].to_numpy()
    cells = pool["cell_id"].to_numpy()
    out = {}
    for hold in holdouts:
        mask = domains == hold
        a = full_features(pool.filter(pl.Series(~mask))).select(cols).to_numpy()
        b = full_features(pool.filter(pl.Series(mask))).select(cols).to_numpy()
        m = fit(model, a, y[~mask], cell_weights(cells[~mask]))
        out[hold] = metrics(y[mask], predict(m, b))
    return out


#: Отложенные домены, по которым выбирается модель. Основная номинация — NCA + NMC,
#: ячейки теста стареют примерно до SOH 0.78. Sandia NCA — химия теста с износом
#: до 0.86; Oxford — другой домен с износом до 0.77.
TRANSFER_HOLDOUTS = ["sandia_NCA", "oxford"]
#: Справочно, в выборе не участвуют:
#: * Sandia NMC — ячейки почти новые (SOH >= 0.98), там выигрывает модель, которая
#:   всем ставит «как новая», то есть ровно та ошибка, которую надо избежать на тесте;
#: * CALCE — без него в обучении почти не остаётся состаренных ячеек, все модели
#:   там хуже константы.
REFERENCE_HOLDOUTS = ["sandia_NMC", "calce"]
#: Признак запоминания: ошибка на незнакомых ячейках (CV по ячейкам) больше
#: ошибки на своих обучающих данных более чем в 1.5 раза.
MAX_MEMORIZATION_RATIO = 1.5


def main() -> None:
    """Сравнить модели, сохранить отчёт и сабмит."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    train = pl.read_parquet(DATA / "train.parquet")
    val = pl.read_parquet(DATA / "val.parquet")
    test = pl.read_parquet(DATA / "test_unlabeled.parquet")
    pool = pl.concat([train, val])

    # Признаки считаются отдельно в каждом наборе (относительные — внутри набора).
    ftr, fva, fte = full_features(train), full_features(val), full_features(test)
    feature_sets = {"базовые": build_features(train.head(1)).columns, "устойчивые": ROBUST_COLS}

    ytr, yva = train["soh"].to_numpy(), val["soh"].to_numpy()
    ctr, cva = train["cell_id"].to_numpy(), val["cell_id"].to_numpy()
    wtr = cell_weights(ctr)
    ds_va = np.array([dataset_of(c) for c in cva])
    pool_ds = np.array([dataset_of(c) for c in pool["cell_id"]])
    pool_lab = np.array([d if d.startswith("sandia") else d.split("_")[0] for d in pool_ds])
    holdouts = TRANSFER_HOLDOUTS + REFERENCE_HOLDOUTS

    summary: dict[str, dict] = {}
    val_preds: dict[str, np.ndarray] = {}
    gkf = GroupKFold(n_splits=5)
    for name, (fs_name, model) in candidates().items():
        cols = feature_sets[fs_name]
        xtr, xva = ftr.select(cols).to_numpy(), fva.select(cols).to_numpy()
        oof = np.zeros_like(ytr)
        for tr_idx, te_idx in gkf.split(xtr, ytr, ctr):
            m = fit(model, xtr[tr_idx], ytr[tr_idx], wtr[tr_idx])
            oof[te_idx] = predict(m, xtr[te_idx])
        m = fit(model, xtr, ytr, wtr)
        in_sample = metrics(ytr, predict(m, xtr))
        cv = metrics(ytr, oof)
        pva = predict(m, xva)
        val_preds[name] = pva
        lodo = lodo_eval(model, pool, cols, pool_lab, holdouts)
        summary[name] = {
            "features": fs_name,
            "train_in_sample": in_sample,
            "cv_train_by_cell": cv,
            "memorization_ratio": round(cv["mae_pp"] / max(in_sample["mae_pp"], 1e-9), 3),
            "val": metrics(yva, pva),
            "val_by_dataset": {
                d: metrics(yva[ds_va == d], pva[ds_va == d]) for d in sorted(set(ds_va))
            },
            "transfer_holdout": lodo,
            "transfer_mae_avg": round(
                float(np.mean([lodo[h]["mae_pp"] for h in TRANSFER_HOLDOUTS])), 3
            ),
        }
        print(f"готово: {name}", flush=True)

    # Выбор: модель не должна запоминать обучающие ячейки (отношение <= 1.5),
    # среди таких — лучший перенос на отложенные домены. Тестовые метки и val
    # (на 90 % это CALCE) в выборе не участвуют.
    allowed = [
        k
        for k, s in summary.items()
        if not k.startswith("mean") and s["memorization_ratio"] <= MAX_MEMORIZATION_RATIO
    ]
    best = min(allowed, key=lambda k: summary[k]["transfer_mae_avg"])
    best_fs, best_model = candidates()[best]
    best_cols = feature_sets[best_fs]

    pbest = val_preds[best]
    pl.DataFrame(
        {
            "fragment_id": val["fragment_id"],
            "cell_id": cva,
            "dataset": ds_va,
            "soh_true": np.round(yva, 5),
            "soh_pred": np.round(pbest, 5),
            "error": np.round(pbest - yva, 5),
            "abs_error": np.round(np.abs(pbest - yva), 5),
        }
    ).write_csv(RESULTS / "val_predictions.csv")

    bins = [0.0, 0.7, 0.8, 0.9, 0.95, 1.2]
    by_soh = {}
    for lo, hi in pairwise(bins):
        mask = (yva >= lo) & (yva < hi)
        if mask.any():
            by_soh[f"[{lo:.2f}, {hi:.2f})"] = metrics(yva[mask], pbest[mask])

    # Финальная модель на train + val (признаки каждого набора — внутри набора).
    xall = np.vstack([ftr.select(best_cols).to_numpy(), fva.select(best_cols).to_numpy()])
    yall = np.concatenate([ytr, yva])
    final = fit(best_model, xall, yall, cell_weights(np.concatenate([ctr, cva])))
    pte = np.round(predict(final, fte.select(best_cols).to_numpy()), 5)
    pl.DataFrame({"fragment_id": test["fragment_id"], "soh_pred": pte}).write_csv(
        OUT / "submission.csv"
    )

    report = {
        "best_model": best,
        "selection": "min transfer_mae_avg among models with memorization_ratio <= 1.5",
        "transfer_holdouts": TRANSFER_HOLDOUTS,
        "reference_holdouts": REFERENCE_HOLDOUTS,
        "features": best_cols,
        "models": summary,
        "best_val_by_soh": by_soh,
        # Признаки train и val (каждый посчитан внутри своего набора), по доменам.
        "feature_soh_correlations": soh_correlations(
            pl.concat([ftr, fva]), pool["soh"].to_numpy(), pool_lab
        ),
        "test_prediction_stats": {
            "n": int(pte.size),
            "mean": round(float(pte.mean()), 4),
            "p10": round(float(np.percentile(pte, 10)), 4),
            "min": float(pte.min()),
            "max": float(pte.max()),
        },
    }
    (RESULTS / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    header = f"\n{'модель':<22}{'train':>7}{'CV':>7}{'CV/train':>9}{'val':>7}"
    print(header + "".join(f"{h:>12}" for h in holdouts) + f"{'перенос':>9}")
    for key, s in summary.items():
        row = (
            f"{key:<22}{s['train_in_sample']['mae_pp']:>7.2f}"
            f"{s['cv_train_by_cell']['mae_pp']:>7.2f}{s['memorization_ratio']:>9.2f}"
            f"{s['val']['mae_pp']:>7.2f}"
        )
        cells = "".join(f"{s['transfer_holdout'][h]['mae_pp']:>12.2f}" for h in holdouts)
        print(row + cells + f"{s['transfer_mae_avg']:>9.2f}")
    print(f"\nлучшая: {best}")
    print("перенос:", json.dumps(summary[best]["transfer_holdout"], ensure_ascii=False))
    print("val по датасетам:", json.dumps(summary[best]["val_by_dataset"], ensure_ascii=False))
    print("val по уровню SOH:", json.dumps(by_soh, ensure_ascii=False))
    print("прогноз test:", report["test_prediction_stats"])
    print("корреляция с SOH:")
    for domain, row in report["feature_soh_correlations"].items():
        print(f"  {domain:<12}", "  ".join(f"{k}={v:+.2f}" for k, v in row.items()))


if __name__ == "__main__":
    main()
