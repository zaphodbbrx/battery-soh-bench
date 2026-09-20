#!/usr/bin/env bash
# Скачивание архива циклирования из R&D Data Repository Sandia (Barkholtz et al.).
#
# Источник: https://www.sandia.gov/ess/tools-resources/rd-data-repository
# Прямая ссылка, без регистрации и запросов.
#
# Роль в бенчмарке: **только обучение**. Это другое исследование, чем SNL/Preger
# (скрытый тест, ТЗ §17.4): иная программа тестирования (2016-2017), другой набор
# ячеек. Важно, что здесь есть NCA, NMC и LFP — те же химии, что в тесте, — а
# обучение без них состоит из LCO (CALCE, Oxford) и упирается в §9.
#
# Состав: 179 файлов .xls в 60 каталогах, 24 ячейки по 6 на химию (LCO, LFP,
# NCA, NMC), температуры 5/15/25/35/45 °C. Формат — Arbin, тот же, что у CALCE,
# плюс каналы температуры, которых у CALCE нет.
#
# Атрибуция: H.M. Barkholtz et al., J. Electrochem. Soc. 164 (2017) A2697,
# DOI 10.1149/2.1701712jes, и R&D Data Repository Sandia.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${1:-$ROOT/data/raw/sandia_rd}"
ARCHIVE_URL="https://www.sandia.gov/app/uploads/sites/163/Sandia_Cell_Cycle_Testing_Data.zip"
ARCHIVE="$DEST/_archives/Sandia_Cell_Cycle_Testing_Data.zip"
#: sha256 архива; зафиксирован после первой загрузки.
EXPECTED_SHA256="7e511831c9e23c3cef0e4a12fee7b4b3a52985f7d3ad770e14ea7c33e18ee323"

mkdir -p "$DEST/_archives"

verify() {
    local got
    got="$(sha256sum "$ARCHIVE" | cut -d' ' -f1)"
    if [[ -z "$EXPECTED_SHA256" ]]; then
        echo "    sha256(не зафиксирован)=$got"
        return 0
    fi
    if [[ "$got" != "$EXPECTED_SHA256" ]]; then
        echo "    ОШИБКА sha256: ожидался $EXPECTED_SHA256, получен $got" >&2
        return 1
    fi
    echo "    sha256 совпал"
}

if [[ -f "$ARCHIVE" ]]; then
    echo "== архив уже есть, проверяю"
    verify || { echo "    перекачиваю" >&2; rm -f "$ARCHIVE"; }
fi

if [[ ! -f "$ARCHIVE" ]]; then
    echo "== скачиваю (312 МБ, R&D Data Repository Sandia)"
    curl --fail --location --retry 3 --retry-delay 5 --silent --show-error \
        --output "$ARCHIVE.part" "$ARCHIVE_URL"
    mv "$ARCHIVE.part" "$ARCHIVE"
    verify
fi

count="$(find "$DEST" -name '*.xls' | wc -l)"
if [[ "$count" -eq 0 ]]; then
    echo "== распаковка"
    unzip -q -o "$ARCHIVE" -d "$DEST"
else
    echo "== распаковано ранее ($count файлов), пропускаю"
fi

echo
echo "готово: $DEST"
echo "файлов .xls: $(find "$DEST" -name '*.xls' | wc -l)"
echo "дальше: bsb ingest --dataset sandia_rd --raw $DEST"
