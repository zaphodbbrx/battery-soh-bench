"""Участническая команда проверки CSV сабмита."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from bsb.config import load_config
from bsb.validate_submission import expected_test_ids, validate_submission

EXIT_OK = 0
EXIT_FAILURE = 1


def main(argv: Sequence[str] | None = None) -> int:
    """Проверить формат и полноту CSV сабмита."""
    parser = argparse.ArgumentParser(prog="bsb")
    parser.add_argument("--config", type=Path, default=None, help="путь к scoring-конфигу")
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate", help="проверить формат сабмита")
    validate.add_argument("--submission", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        report = validate_submission(
            args.submission, expected_ids=expected_test_ids(config), config=config
        )
    except (OSError, ValueError) as exc:
        print(f"ошибка: {exc}", file=sys.stderr)
        return EXIT_FAILURE
    print(report.render())
    return EXIT_OK if report.ok else EXIT_FAILURE
