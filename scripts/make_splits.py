#!/usr/bin/env python
"""Генерация ``splits/*.txt`` (ТЗ §8).

Списки коммитятся и создаются один раз. Правило отбора val лежит в конфиге
(``splits.val_rule``), тест — датасет целиком; train получает всё остальное.

Проверка утечки выполняется здесь же: и по ячейкам, и по партиям (``batch_id``
означает физическую ячейку, которая в архиве Sandia тестировалась при
нескольких температурах, поэтому её записи обязаны попасть в один сплит).

Запуск::

    python scripts/make_splits.py            # показать, что получится
    python scripts/make_splits.py --write    # записать файлы
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bsb.config import BenchmarkConfig, load_config
from bsb.splits import assert_no_cell_leakage, split_path, write_split
from bsb.store import read_cells

HEADERS = {
    "train": (
        "Сплит train (ТЗ §8): по одному cell_id в строке.\n"
        "\n"
        "Генерируется scripts/make_splits.py, правится только владельцем.\n"
        "Единица разбиения — ячейка, при известной партии — партия (§8)."
    ),
    "val": (
        "Сплит val (ТЗ §8): по одному cell_id в строке.\n"
        "\n"
        "Собран правилом splits.val_rule: целые ячейки CALCE и Oxford\n"
        "(состаренный LCO) плюс целые физические ячейки NCA из Barkholtz\n"
        "(тестовая химия, но свежие). Вместе покрывают обе оси переноса,\n"
        "по отдельности — ни одна.\n"
        "Генерируется scripts/make_splits.py."
    ),
    "test": (
        "Сплит test (ТЗ §8): по одному cell_id в строке.\n"
        "\n"
        "Источник — SNL из BatteryLife (CC-BY-4.0), см. README «Источник теста».\n"
        "Список используется тулкитом для генерации эталона: участникам\n"
        "соответствие fragment_id -> ячейка не выдаётся."
    ),
}


def build_splits(config: BenchmarkConfig) -> dict[str, list[str]]:
    """Собрать train / val / test по конфигу.

    Raises
    ------
    SystemExit
        Если ячейка из правила val отсутствует в хранилище — это опечатка в
        конфиге, и молча получить другую валидацию нельзя.
    """
    cells = read_cells(config)
    available = set(cells["cell_id"].to_list())
    rule = config.splits.val_rule

    test = set(cells.filter(pl.col("dataset").is_in(list(config.splits.test_datasets)))["cell_id"])
    val = set(rule.all_cells)
    # Из Barkholtz берём целые физические ячейки: у них batch_id — это имя
    # ячейки, а записи одной ячейки отличаются температурой и режимом.
    val |= set(
        cells.filter(
            (pl.col("dataset") == "sandia_rd")
            & pl.col("batch_id").is_in(list(rule.sandia_rd_cells))
        )["cell_id"]
    )
    missing = sorted(val - available)
    if missing:
        raise SystemExit(f"правило val ссылается на ячейки, которых нет в хранилище: {missing[:5]}")
    # train ограничен splits.train_datasets: без этого переключатель A/B
    # («с Barkholtz и без») не действовал бы, и датасет попадал бы в обучение
    # независимо от конфига.
    train = set(
        cells.filter(pl.col("dataset").is_in(list(config.splits.train_datasets)))["cell_id"]
    )
    train -= val | test
    if not train:
        raise SystemExit("train пуст: правило val забрало все ячейки обучения")
    if not val <= set(
        cells.filter(pl.col("dataset").is_in(list(config.splits.val_datasets)))["cell_id"]
    ):
        raise SystemExit("val содержит ячейки датасетов, которых нет в splits.val_datasets")
    return {"train": sorted(train), "val": sorted(val), "test": sorted(test)}


def main() -> int:
    """Собрать (и при ``--write`` записать) списки сплитов."""
    parser = argparse.ArgumentParser(description="Генерация splits/*.txt (ТЗ §8)")
    parser.add_argument("--write", action="store_true", help="записать файлы")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    splits = build_splits(config)
    cells = read_cells(config)
    batches = {
        row["cell_id"]: str(row["batch_id"])
        for row in cells.filter(pl.col("batch_id").is_not_null()).iter_rows(named=True)
    }
    assert_no_cell_leakage(splits, cell_to_batch=batches or None)

    for name, identifiers in splits.items():
        dataset_counts = (
            cells.filter(pl.col("cell_id").is_in(identifiers))
            .group_by("dataset")
            .len()
            .sort("dataset")
            .to_dicts()
        )
        chemistry = (
            cells.filter(pl.col("cell_id").is_in(identifiers))
            .group_by("chemistry")
            .len()
            .sort("chemistry")
            .to_dicts()
        )
        print(f"{name:>5}: {len(identifiers):4d} ячеек")
        print(f"        по датасетам: {dataset_counts}")
        print(f"        по химиям:    {chemistry}")
    print("\nутечки между сплитами нет (проверено и по партиям)")

    if args.write:
        for name, identifiers in splits.items():
            path = split_path(config, name)
            write_split(path, identifiers, header=HEADERS[name])
            print(f"записано: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
