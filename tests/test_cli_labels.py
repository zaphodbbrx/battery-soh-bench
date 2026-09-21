"""Тесты команды ``labels`` (ТЗ §13, §5).

Команда пересчитывает опорные циклы и SOH по уже разобранному ``timeseries``,
не перечитывая сырые файлы. Главное свойство, которое здесь проверяется, —
идемпотентность: повторная разметка на тех же правилах обязана дать те же
метки, иначе разделение «разбор / разметка» бессмысленно.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl
import yaml

from bsb.cli import EXIT_FAILURE, EXIT_OK, main
from bsb.config import BenchmarkConfig, load_config
from bsb.ingest import get_adapter
from bsb.ingest.base import rebuild_raw_cycles
from bsb.store import compact, read_cycles, write_bundle
from synthetic import synth_raw_bundle

REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_config(tmp_path: Path, overrides: dict[str, Any] | None = None) -> Path:
    """Записать копию репозиторного конфига в ``tmp_path/configs``.

    Кладём именно в ``configs/``, потому что корень репозитория выводится как
    ``config_path.parent.parent``: иначе относительные пути данных ушли бы не
    в ``tmp_path``.
    """
    data = yaml.safe_load((REPO_ROOT / "configs" / "benchmark.yaml").read_text(encoding="utf-8"))
    for dotted, value in (overrides or {}).items():
        node = data
        *parents, leaf = dotted.split(".")
        for key in parents:
            node = node[key]
        node[leaf] = value
    target = tmp_path / "configs" / "benchmark.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(data), encoding="utf-8")
    return target


def _prepared_store(tmp_path: Path, dataset: str = "oxford") -> tuple[Path, BenchmarkConfig]:
    """Собрать хранилище с одной синтетической ячейкой и вернуть путь к конфигу.

    Ёмкости берём не из синтетического кадра, а восстанавливаем из ``timeseries``
    (:func:`~bsb.ingest.base.rebuild_raw_cycles`): иначе фикстура была бы
    самопротиворечивой — её ёмкости не совпадали бы с её же кривыми, и проверка
    идемпотентности ловила бы этот артефакт вместо поведения команды. На
    реальных данных такого расхождения нет: из 13 329 опорных циклов четырёх
    датасетов ёмкость совпадает с сохранённой до последнего знака.
    """
    config_path = _write_config(tmp_path)
    config = load_config(config_path)
    raw = synth_raw_bundle(dataset=dataset)
    cycles = rebuild_raw_cycles(raw.cell_id, raw.timeseries)
    prepared = get_adapter(dataset, config).mark_reference_cycles(
        type(raw)(cells=raw.cells, cycles=cycles, timeseries=raw.timeseries)
    )
    write_bundle(config, prepared, dataset=dataset)
    compact(config, datasets=[dataset], allow_partial=True)
    return config_path, config


def _labels(config: BenchmarkConfig) -> pl.DataFrame:
    cycles = read_cycles(config)
    return cycles.select("cell_id", "cycle_index", "is_reference", "soh", "soh_source").sort(
        "cell_id", "cycle_index"
    )


def test_labels_reproduces_stored_labels(tmp_path: Path) -> None:
    """Повторная разметка теми же правилами не меняет ни одного цикла."""
    config_path, config = _prepared_store(tmp_path)
    before = _labels(config)
    assert before.filter(pl.col("is_reference")).height > 0

    code = main(["--config", str(config_path), "labels", "--dataset", "oxford", "--compact"])

    assert code == EXIT_OK
    assert _labels(config).equals(before)


def test_labels_recomputes_from_timeseries_not_from_store(tmp_path: Path) -> None:
    """Метки берутся из правила заново, а не копируются из хранилища.

    Портим ``is_reference`` прямо в унифицированном уровне: если бы команда
    переписывала кадр как есть, метки остались бы снятыми.
    """
    from bsb.dataset import load_bundle

    config_path, config = _prepared_store(tmp_path)
    cell_id = read_cycles(config)["cell_id"][0]
    bundle = load_bundle(config, cell_id)
    spoilt = type(bundle)(
        cells=bundle.cells,
        cycles=bundle.cycles.with_columns(pl.lit(False).alias("is_reference")),
        timeseries=bundle.timeseries,
    )
    write_bundle(config, spoilt, dataset="oxford")
    compact(config, datasets=["oxford"], allow_partial=True)
    assert _labels(config).filter(pl.col("is_reference")).height == 0

    code = main(["--config", str(config_path), "labels", "--dataset", "oxford", "--compact"])

    assert code == EXIT_OK
    assert _labels(config).filter(pl.col("is_reference")).height > 0


def test_labels_reports_unknown_dataset(tmp_path: Path) -> None:
    """Пустой датасет — это понятная ошибка с подсказкой, а не падение."""
    config_path, _ = _prepared_store(tmp_path)

    code = main(["--config", str(config_path), "labels", "--dataset", "calce"])

    assert code == EXIT_FAILURE
