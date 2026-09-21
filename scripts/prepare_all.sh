#!/usr/bin/env bash
# Полная подготовка данных: ingest всех датасетов -> compact -> фрагменты.
#
# Требует, чтобы сырые данные уже лежали в data/raw/<dataset>
# (см. scripts/download_*.sh). Скрипт идемпотентен: повторный ingest ячейки
# заменяет её строки, а не дублирует их.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Окружение: если в проекте есть uv, он и запускает команды — тогда версии
# ровно те, что в uv.lock. Иначе берём интерпретатор из .venv (pip-путь).
# Переменная PYTHON позволяет указать свой интерпретатор явно.
if [[ -n "${PYTHON:-}" ]]; then
    BSB=("$PYTHON" -m bsb.cli)
elif command -v uv >/dev/null 2>&1; then
    BSB=(uv run --quiet python -m bsb.cli)
else
    PYTHON="$ROOT/.venv/bin/python"
    BSB=("$PYTHON" -m bsb.cli)
fi

if [[ -z "${PYTHON:-}" ]] && ! command -v uv >/dev/null 2>&1 && [[ ! -x "$ROOT/.venv/bin/python" ]]; then
    echo "нет окружения: установите uv (https://docs.astral.sh/uv/) и выполните" >&2
    echo "    uv sync" >&2
    echo "либо создайте окружение вручную:" >&2
    echo "    python -m venv .venv && .venv/bin/pip install -e '.[dev]'" >&2
    exit 1
fi

# Порядок — по плану §15: Oxford первым (сквозная проверка пайплайна).
# sandia_rd обязателен: он даёт обучению химии теста (NCA, NMC, LFP), и без него
# основная номинация на бейзлайне падает с 4.79 до 9.35 п.п. (см.
# docs/04-ориентиры.md). Наборы — те, что перечислены в
# splits.train_datasets / val_datasets / test_datasets.
DATASETS=(oxford calce sandia_rd sandia kit)

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
