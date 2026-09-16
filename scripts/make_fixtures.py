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
CALCE_FIXTURES = {
    "CS2_33/CS2_33/CS2_33_10_05_10.xlsx": "calce_CS2_33_arbin.xlsx",
    "CS2_8/CS2_8/CS2_8_1_28_10.txt": "calce_CS2_8_cadex.txt",
}


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
