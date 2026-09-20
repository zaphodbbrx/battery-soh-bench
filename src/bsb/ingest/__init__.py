"""Адаптеры датасетов (ТЗ §6).

Реестр :data:`ADAPTERS` — единственная точка получения адаптера по имени,
поэтому CLI не знает ничего датасет-специфичного.
"""

from __future__ import annotations

from collections.abc import Callable

from bsb.config import BenchmarkConfig
from bsb.ingest.base import CellSource, DatasetAdapter
from bsb.ingest.calce import CalceAdapter
from bsb.ingest.kit import KitAdapter
from bsb.ingest.oxford import OxfordAdapter
from bsb.ingest.sandia import SandiaAdapter
from bsb.ingest.sandia_rd import SandiaRdAdapter

__all__ = [
    "ADAPTERS",
    "CalceAdapter",
    "CellSource",
    "DatasetAdapter",
    "KitAdapter",
    "OxfordAdapter",
    "SandiaAdapter",
    "SandiaRdAdapter",
    "get_adapter",
]

ADAPTERS: dict[str, Callable[[BenchmarkConfig], DatasetAdapter]] = {
    OxfordAdapter.name: OxfordAdapter,
    CalceAdapter.name: CalceAdapter,
    KitAdapter.name: KitAdapter,
    SandiaAdapter.name: SandiaAdapter,
    SandiaRdAdapter.name: SandiaRdAdapter,
}


def get_adapter(name: str, config: BenchmarkConfig) -> DatasetAdapter:
    """Вернуть адаптер по имени датасета.

    Raises
    ------
    KeyError
        Если датасет неизвестен.
    """
    if name not in ADAPTERS:
        raise KeyError(f"неизвестный датасет {name!r}; известные: {sorted(ADAPTERS)}")
    return ADAPTERS[name](config)
