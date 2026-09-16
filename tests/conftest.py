"""Общие фикстуры тестов."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

from bsb.config import BenchmarkConfig, load_config, load_config_from_string
from bsb.schema import CellBundle
from synthetic import synth_bundle

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def repo_config() -> BenchmarkConfig:
    """Конфиг репозитория как есть."""
    return load_config()


@pytest.fixture
def make_config(tmp_path: Path) -> Callable[[dict[str, Any] | None], BenchmarkConfig]:
    """Фабрика конфигов: копия репозиторного YAML с точечными правками.

    Ключи правок — точечные пути, например ``{"fragment.delta_v": 0.5}``.
    Пути при этом указывают в ``tmp_path``, поэтому тесты пишут данные в
    временный каталог и не трогают рабочий ``data/``.
    """

    def _make(overrides: dict[str, Any] | None = None) -> BenchmarkConfig:
        data = yaml.safe_load(
            (REPO_ROOT / "configs" / "benchmark.yaml").read_text(encoding="utf-8")
        )
        for dotted, value in (overrides or {}).items():
            node = data
            *parents, leaf = dotted.split(".")
            for key in parents:
                node = node[key]
            node[leaf] = value
        return load_config_from_string(yaml.safe_dump(data), repo_root=tmp_path)

    return _make


@pytest.fixture
def synthetic_bundle() -> CellBundle:
    """Готовый синтетический бандл на 6 циклов."""
    return synth_bundle()
