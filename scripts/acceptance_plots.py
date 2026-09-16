#!/usr/bin/env python
"""Приёмка адаптера глазами (ТЗ §14).

Строит график ``soh`` по циклам для 3-5 ячеек датасета и печатает таблицу с
механическими признаками поломки разметки: немонотонности и «ступеньки».

Кривая SOH обязана монотонно убывать без ступенек и выбросов. Ступенька
означает ошибку в определении опорных циклов, и никакой юнит-тест её не
поймает — поэтому числа здесь вспомогательные, а решение принимается глазами.

Запуск::

    python scripts/acceptance_plots.py --dataset oxford --cells 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bsb.config import load_config
from bsb.store import read_cells, read_cycles

#: Падение SOH более чем на эту величину между соседними опорными циклами
#: считается подозрительной «ступенькой» и требует взгляда человека.
STEP_THRESHOLD = 0.05


def _trajectory_checks(cycles: pl.DataFrame) -> dict[str, float | int | str]:
    """Механические признаки для одной ячейки."""
    frame = cycles.sort("cycle_index")
    soh = frame["soh"].to_numpy()
    index = frame["cycle_index"].to_numpy()
    deltas = soh[1:] - soh[:-1]
    worst_up = float(deltas.max()) if deltas.size else 0.0
    worst_down = float(-deltas.min()) if deltas.size else 0.0
    step_at = int(index[1:][deltas.argmin()]) if deltas.size else 0
    return {
        "n": frame.height,
        "soh_first": round(float(soh[0]), 4),
        "soh_last": round(float(soh[-1]), 4),
        "max_recovery": round(worst_up, 4),
        "max_step": round(worst_down, 4),
        "step_at_cycle": step_at,
        "monotone": bool((deltas <= 1e-9).all()),
        "suspicious_steps": int((worst_down > STEP_THRESHOLD) * 1),
    }


def main() -> int:
    """Построить графики приёмки и напечатать таблицу проверок."""
    parser = argparse.ArgumentParser(description="Приёмка адаптера по графикам SOH (ТЗ §14)")
    parser.add_argument("--dataset", default="oxford")
    parser.add_argument("--cells", type=int, default=5, help="сколько ячеек показать (3-5)")
    parser.add_argument("--out", type=Path, default=None, help="куда положить PNG")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    cells = read_cells(config, datasets=[args.dataset])
    if cells.height == 0:
        print(f"нет ячеек датасета {args.dataset} в унифицированном уровне", file=sys.stderr)
        return 1
    cycles = read_cycles(config, cell_ids=cells["cell_id"].to_list())

    # Длинные траектории информативнее: берём ячейки с наибольшим числом циклов.
    counts = (
        cycles.group_by("cell_id")
        .len()
        .sort("len", descending=True)
        .head(args.cells)["cell_id"]
        .to_list()
    )
    print(f"датасет {args.dataset}: {cells.height} ячеек, показано {len(counts)}")
    header = (
        f"{'cell_id':>16} {'n':>4} {'soh_first':>9} {'soh_last':>9} "
        f"{'max_recovery':>12} {'max_step':>9} {'step@cycle':>11} {'monotone':>9}"
    )
    print(header)
    print("-" * len(header))
    for cell_id in counts:
        checks = _trajectory_checks(cycles.filter(pl.col("cell_id") == cell_id))
        print(
            f"{cell_id:>16} {checks['n']:>4} {checks['soh_first']:>9} {checks['soh_last']:>9} "
            f"{checks['max_recovery']:>12} {checks['max_step']:>9} {checks['step_at_cycle']:>11} "
            f"{checks['monotone']!s:>9}"
            + ("   <-- СТУПЕНЬКА" if checks["suspicious_steps"] else "")
        )

    figure, axes = plt.subplots(
        len(counts), 1, figsize=(9, 2.2 * len(counts)), sharex=False, squeeze=False
    )
    for axis, cell_id in zip(axes[:, 0], counts, strict=True):
        frame = cycles.filter(pl.col("cell_id") == cell_id).sort("cycle_index")
        axis.plot(frame["cycle_index"], frame["soh"], marker="o", markersize=3, linewidth=1)
        axis.set_ylabel("SOH")
        axis.set_title(f"{cell_id}  ({frame.height} опорных измерений)", fontsize=10)
        axis.grid(alpha=0.3)
    axes[-1, 0].set_xlabel("cycle_index")
    figure.suptitle(f"Приёмка адаптера {args.dataset}: SOH по циклам (ТЗ §14)", fontsize=12)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    out_path = args.out or (config.root / "reports" / f"acceptance_{args.dataset}.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=130)
    print(f"\nграфик: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
