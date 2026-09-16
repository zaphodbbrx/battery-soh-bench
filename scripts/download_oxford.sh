#!/usr/bin/env bash
# Скачивание Oxford Battery Degradation Dataset 1 (ТЗ §17.3).
#
# Источник: ORA, University of Oxford, DOI 10.5287/bodleian:KO2kdmYGg
# Лицензия: ODbL 1.0 (содержимое — DbCL 1.0). Цитировать набор и диссертацию:
#   Christoph R. Birkl, "Diagnosis and Prognosis of Degradation in Lithium-Ion
#   Batteries", PhD thesis, University of Oxford, 2017.
#
# В наборе три файла:
#   Readme.txt                            — описание эксперимента
#   ExampleDC_C1.mat                      — один первый цикл целиком
#   Oxford_Battery_Degradation_Dataset_1.mat — 253.8 МБ, ТОЛЬКО характеризация
#
# Скрипт идемпотентен: уже скачанный файл с совпавшим sha256 не перекачивается.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${1:-$ROOT/data/raw/oxford}"
BASE="https://ora.ox.ac.uk/objects/uuid:03ba4b01-cfed-46d3-9b1a-7d4a7bdf6fac/files"

mkdir -p "$DEST"

files=(
    "Readme.txt|$BASE/m43cc05e7c5f1245f4895d9dbd495e52f"
    "ExampleDC_C1.mat|$BASE/me9fc40a60ac98708f1b73f3a836548e9"
    "Oxford_Battery_Degradation_Dataset_1.mat|$BASE/m5ac36a1e2073852e4f1f7dee647909a7"
)

# sha256 зафиксированы после первой загрузки; пустая строка = ещё не зафиксирован.
declare -A expected=(
    ["Readme.txt"]="f04c8bdae97e98632b9daeaa7c71631990dbf5ecbf38711347fc470baefc0def"
    ["ExampleDC_C1.mat"]="144cdf5aec7467bcce5445e132495e8477f4e08d9963c221d8ba5d7970a7a97d"
    ["Oxford_Battery_Degradation_Dataset_1.mat"]="a8f0b928f4ea77d7ba98b97194b78cc77336f2338fee8b57573d043f0cf26781"
)

verify() {
    local path="$1" want="$2"
    local got
    got="$(sha256sum "$path" | cut -d' ' -f1)"
    if [[ -z "$want" ]]; then
        echo "    sha256(не зафиксирован)=$got"
        return 0
    fi
    if [[ "$got" != "$want" ]]; then
        echo "    ОШИБКА sha256: ожидался $want, получен $got" >&2
        return 1
    fi
    echo "    sha256 совпал"
}

for entry in "${files[@]}"; do
    name="${entry%%|*}"
    url="${entry#*|}"
    target="$DEST/$name"
    want="${expected[$name]:-}"
    if [[ -f "$target" ]]; then
        echo "== $name: уже есть, проверяю"
        if verify "$target" "$want"; then
            continue
        fi
        echo "    перекачиваю" >&2
    else
        echo "== $name: скачиваю"
    fi
    curl --fail --location --retry 3 --retry-delay 5 --progress-bar \
        --output "$target.part" "$url"
    mv "$target.part" "$target"
    verify "$target" "$want"
done

echo
echo "готово: $DEST"
ls -la "$DEST"
