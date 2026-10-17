"""Экспортировать закрытый эталон в компактный GitHub Actions Secret.

Запускать только организатору из корня репозитория. Содержимое вывода —
конфиденциальные метки теста; файл по умолчанию лежит внутри игнорируемого data/.
"""

from __future__ import annotations

import argparse
import base64
import gzip
from pathlib import Path

from bsb.config import load_config
from bsb.eval import load_truth

MAX_SECRET_CHARS = 48_000
DEFAULT_OUTPUT = Path("data/private/test_truth.b85")


def main() -> int:
    """Сформировать файл для ручной загрузки командой gh secret set."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    truth = load_truth(load_config()).select(
        "fragment_id", "soh", "chemistry", "temperature_setpoint_c"
    )
    csv_bytes = truth.write_csv().encode("utf-8")
    payload = base64.b85encode(gzip.compress(csv_bytes, mtime=0)).decode("ascii")
    if len(payload) > MAX_SECRET_CHARS:
        raise SystemExit(
            f"Эталон занимает {len(payload)} символов, лимит экспортёра — {MAX_SECRET_CHARS}. "
            "Нужен внешний приватный источник данных вместо одного Actions Secret."
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(payload, encoding="ascii")
    args.out.chmod(0o600)
    print(
        f"Подготовлен секрет: {args.out} ({len(payload)} символов, "
        f"{truth.height} строк). Загрузите его через gh secret set BSB_TEST_TRUTH_B85."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
