"""battery-soh-bench — тулкит соревнования по оценке SOH литий-ионной ячейки.

Публичный API пакета намеренно узкий: конфиг, схема §4, хранилище, сплиты и
CLI. Всё остальное (адаптеры, генератор фрагментов, бейзлайны) живёт в
подмодулях и вызывается через ``bsb``.
"""

from __future__ import annotations

from bsb.config import BenchmarkConfig, ConfigError, load_config
from bsb.schema import (
    CellBundle,
    SchemaViolation,
    validate_bundle,
    validate_cells,
    validate_cycles,
    validate_fragments,
    validate_timeseries,
)
from bsb.splits import (
    CellLeakageError,
    assert_no_cell_leakage,
    load_splits,
    read_split,
    write_split,
)

__version__ = "0.1.0"

__all__ = [
    "BenchmarkConfig",
    "CellBundle",
    "CellLeakageError",
    "ConfigError",
    "SchemaViolation",
    "__version__",
    "assert_no_cell_leakage",
    "load_config",
    "load_splits",
    "read_split",
    "validate_bundle",
    "validate_cells",
    "validate_cycles",
    "validate_fragments",
    "validate_timeseries",
    "write_split",
]
