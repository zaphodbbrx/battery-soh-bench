#!/usr/bin/env bash
# Полная подготовка данных: ingest всех датасетов -> compact -> фрагменты.
#
# Требует, чтобы сырые данные уже лежали в data/raw/<dataset>
# (см. scripts/download_*.sh). Скрипт идемпотентен: повторный ingest ячейки
# заменяет её строки, а не дублирует их.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
BSB=("$PYTHON" -m bsb.cli)

if [[ ! -x "$PYTHON" ]]; then
    echo "нет интерпретатора $PYTHON — создайте окружение: python -m venv .venv && .venv/bin/pip install -e '.[dev]'" >&2
    exit 1
fi

# Порядок — по плану §15: Oxford первым (сквозная проверка пайплайна).
DATASETS=(oxford calce kit sandia)

for dataset in "${DATASETS[@]}"; do
    raw="data/raw/$dataset"
    if [[ ! -d "$raw" ]]; then
        echo "== пропуск $dataset: нет каталога $raw" >&2
        continue
    fi
    echo "== ingest $dataset"
    "${BSB[@]}" ingest --dataset "$dataset" --raw "$raw"
done

echo "== compact"
"${BSB[@]}" compact

for split in train val test; do
    echo "== fragments --split $split"
    "${BSB[@]}" fragments --split "$split" --mode fixed
done

echo "== splits-check"
"${BSB[@]}" splits-check

echo "готово"
