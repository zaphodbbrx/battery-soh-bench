#!/usr/bin/env python
"""Сборка маленьких фикстур из сырых данных для тестов (ТЗ §14).

ТЗ §14 требует smoke-тест на одной реальной ячейке, **зафиксированной в
репозитории**. Сырые наборы весят сотни мегабайт и в git не попадают, поэтому
здесь из ``Oxford_Battery_Degradation_Dataset_1.mat`` вырезается несколько
характеризационных циклов Cell1 и сохраняется сжатый ``.mat`` в ``tests/data``.

Выбор циклов не случаен:

* ``cyc0000`` — начало жизни (bol этой ячейки);
* ``cyc3200`` — аномалия: шаг содержит два склеенных сегмента записи;
* ``cyc8200`` — конец жизни (минимальный SOH в наборе).

Запуск (нужны сырые данные)::

    python scripts/make_fixtures.py
"""

from __future__ import annotations

import argparse
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bsb.config import load_config

#: Ячейка и характеризационные циклы, попадающие в фикстуру.
CELL_FIELD = "Cell1"
CYCLE_SUFFIXES = (0, 3200, 8200)
STEPS = ("C1ch", "C1dc", "OCVch", "OCVdc")
ARRAYS = ("t", "v", "q", "T")

#: Файлы CALCE копируются в фикстуры как есть: один Arbin (Excel) и один CADEX
#: (txt). Они уже небольшие, вырезать из них нечего.
#: Файл архива Sandia R&D для фикстуры. Выбран самый маленький файл, который
#: даёт больше одного опорного цикла: файлы 45 °C содержат только частичные
#: циклы и адаптер их обоснованно пропускает, поэтому «просто самый маленький»
#: не годится.
SANDIA_RD_SOURCE = "Sandia Cell Cycle Testing Data/NMC_1_5C-15C/NMC_1_15C_Mod.xls"
SANDIA_RD_FIXTURE = "sandia_rd_NMC_sample.xls"

CALCE_FIXTURES = {
    "CS2_33/CS2_33/CS2_33_10_05_10.xlsx": "calce_CS2_33_arbin.xlsx",
    "CS2_8/CS2_8/CS2_8_1_28_10.txt": "calce_CS2_8_cadex.txt",
}

#: Ячейка SNL для фикстуры Sandia и сколько циклов из неё оставить.
SANDIA_CELL = "SNL_18650_NMC_25C_0-100_0.5-1C_a.pkl"
SANDIA_CYCLES = 6
SANDIA_FIXTURE = "sandia_SNL_18650_NMC_25C_0-100_0.5-1C_a_small.pkl"


def sandia_window(cycles: list[dict[str, object]], size: int) -> list[dict[str, object]]:
    """Окно циклов вокруг первого capacity check, а не первые попавшиеся.

    У ячейки 0.5-1C рабочие циклы разряжаются на 1C, а проверки ёмкости — на
    0.5C, и идут они не с начала файла. Без этого фикстура не содержала бы ни
    одного опорного цикла и smoke-тест проверял бы не то.
    """
    for index, cycle in enumerate(cycles):
        current = np.asarray(cycle["current_in_A"], dtype=float)
        charge, discharge = current > 0.05, current < -0.05
        if not (charge.any() and discharge.any()):
            continue
        charge_current = abs(float(np.median(current[charge])))
        discharge_current = abs(float(np.median(current[discharge])))
        if charge_current > 0 and abs(discharge_current - charge_current) <= 0.02 * charge_current:
            start = max(0, index - 2)
            return cycles[start : start + size]
    return cycles[:size]


def main() -> int:
    """Вырезать фикстуру из сырого ``.mat``."""
    parser = argparse.ArgumentParser(description="Сборка тестовой фикстуры Oxford (§14)")
    parser.add_argument("--raw", type=Path, default=None, help="каталог data/raw/oxford")
    parser.add_argument("--out", type=Path, default=None, help="куда положить фикстуру")
    args = parser.parse_args()

    config = load_config()
    raw_dir = args.raw or (config.paths.raw_dir / "oxford")
    source = max(raw_dir.glob("*.mat"), key=lambda path: path.stat().st_size)
    out_path = args.out or (config.root / "tests" / "data" / "oxford_Cell1_fixture.mat")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cell = loadmat(source, squeeze_me=True, struct_as_record=False, variable_names=[CELL_FIELD])[
        CELL_FIELD
    ]
    payload: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for suffix in CYCLE_SUFFIXES:
        field = f"cyc{suffix:04d}"
        if not hasattr(cell, field):
            print(f"{field} отсутствует в {source}", file=sys.stderr)
            return 1
        cycle = getattr(cell, field)
        payload[field] = {
            step: {name: np.asarray(getattr(getattr(cycle, step), name)) for name in ARRAYS}
            for step in STEPS
        }
    savemat(out_path, {CELL_FIELD: payload}, do_compression=True, oned_as="row")
    print(f"фикстура: {out_path} ({out_path.stat().st_size / 1024:.0f} КиБ)")
    for field in payload:
        print(
            f"  {field}: " + ", ".join(f"{step}={payload[field][step]['t'].size}" for step in STEPS)
        )

    # Фикстура Sandia: метаданные плюс несколько циклов. Полный pickle весит
    # 18 МБ (тысячи циклов), в репозитории ему не место, а для smoke-теста
    # достаточно первых циклов — среди них есть capacity check.
    sandia_source = config.paths.raw_dir / "sandia" / SANDIA_CELL
    if sandia_source.is_file():
        with sandia_source.open("rb") as handle:
            payload = pickle.load(handle)
        payload = dict(payload)
        payload["cycle_data"] = sandia_window(payload["cycle_data"], SANDIA_CYCLES)
        target = out_path.parent / SANDIA_FIXTURE
        with target.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(
            f"фикстура: {target} ({target.stat().st_size / 1024:.0f} КиБ, "
            f"{len(payload['cycle_data'])} циклов)"
        )
    else:
        print(f"  пропуск фикстуры Sandia: нет {sandia_source}", file=sys.stderr)

    rd_source = config.paths.raw_dir / "sandia_rd" / SANDIA_RD_SOURCE
    if rd_source.is_file():
        target = out_path.parent / SANDIA_RD_FIXTURE
        shutil.copy2(rd_source, target)
        print(f"фикстура: {target} ({target.stat().st_size / 1024:.0f} КиБ, из {rd_source.name})")
    else:
        print(f"  пропуск фикстуры Sandia R&D: нет {rd_source}", file=sys.stderr)

    calce_raw = config.paths.raw_dir / "calce"
    for relative, target_name in CALCE_FIXTURES.items():
        source = calce_raw / relative
        if not source.is_file():
            print(f"  пропуск CALCE-фикстуры: нет {source}", file=sys.stderr)
            continue
        target = out_path.parent / target_name
        shutil.copy2(source, target)
        print(f"фикстура: {target} ({target.stat().st_size / 1024:.0f} КиБ)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
