#!/usr/bin/env bash
# Скачивание серии CALCE CS2 (ТЗ §6.2, §17.1).
#
# Источник: CALCE Battery Data Archive, University of Maryland
#   https://web.calce.umd.edu/batteries/data/
# Цитировать: CALCE battery data base + минимум 3 статьи CALCE (см. страницу).
#
# Все ячейки CS2 заряжаются одинаково (CC-CV 0.5C до 4.2 В, отсечка 0.05 А) —
# именно постоянство заряда делает серию пригодной для бенчмарка (§6.2).
# Различается только разряд: типы 1-6, см. таблицу ниже.
#
# CS2_8 и CS2_21 тестировались на CADEX и лежат в .txt, остальные — Excel
# (Arbin). Это первое, на чём спотыкается адаптер (§17.1).
#
# Скрипт идемпотентен: архив с совпавшим sha256 не перекачивается, распаковка
# пропускается, если каталог ячейки уже непуст.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${1:-$ROOT/data/raw/calce}"
BASE="https://web.calce.umd.edu/batteries/data"

mkdir -p "$DEST/_archives"

# ячейка|тип разряда
cells=(
    "CS2_8|тип 1: CC 0.5C, CADEX .txt"
    "CS2_21|тип 1: CC 0.5C, CADEX .txt"
    "CS2_33|тип 1: CC 0.5C"
    "CS2_34|тип 1: CC 0.5C"
    "CS2_35|тип 2: CC 1C"
    "CS2_36|тип 2: CC 1C"
    "CS2_37|тип 2: CC 1C"
    "CS2_38|тип 2: CC 1C"
    "CS2_3|тип 3: чередование токов разряда 0.11-2.2 А"
    "CS2_9|тип 3: чередование токов разряда 0.11-2.2 А"
    "CS2_7|тип 4: случайная отсечка по напряжению"
    "CS2_5|тип 5: частичные циклы 3.77-2.7 В"
    "CS2_6|тип 5: частичные циклы 3.77-2.7 В"
    "CS2_24|тип 6: частичные циклы 4.2-3.77 В"
    "CS2_25|тип 6: частичные циклы 4.2-3.77 В"
)

# sha256 фиксируются после первой загрузки; пустая строка = ещё не зафиксирован.
declare -A expected=(
    ["CS2_3"]="375f6bb8e6bb8415fb01ea2fb54430f932d13b05fce89721dee77f34999109cd" ["CS2_5"]="717a0c76e308dbb8cbffd217be087437d0484b88845b741bd175204cbc70c7f8" ["CS2_6"]="41e8b0c5dc0509ea9c05a80f51b9949ed9b3bca4bcbeca1cae69d5f9b08760ed" ["CS2_7"]="484fda436972139bf3439b388cbbb64b47c21595e46edcdf5ae8304459caa9c8" ["CS2_8"]="3abf380169cd4afc46c944dc6496f2784465333e710161701515da3d3033e9ae"
    ["CS2_9"]="68367cf84f8486f94d6d2a638b3f421dcce1be7eaaae25464662ad04c49e7914" ["CS2_21"]="28f998ffaee0de92d5b7554dfe5dccf51ef33ae1c3ae526f8b32f1d891c4deb4" ["CS2_24"]="8bb7832b52c627247d8a21cc8b9833328eff5897414645a1f5699b1cec50fd9a" ["CS2_25"]="6c345636f84b019c9054c85a169187ee81de8eeee00fae19b37ff00f17474965" ["CS2_33"]="ce375e3826e840ff0986ceade0d324ba6515f0446697c947353aca097f232505"
    ["CS2_34"]="229dcdc836f140f746c26316569c357b346994fddd3be1223b1e457805c484b4" ["CS2_35"]="84d88cb98a43c02049f16c78806a78ec19084776ea5df8a4b7c2f8f7ba4589e5" ["CS2_36"]="2e674f58bb6dc005c73bef10e87edbbd97afd041dde77a1d461a3c4ecac1b027" ["CS2_37"]="8ed8aecbcbea2bcb17f455923a974be01ea21d7389210e1912d177de5b6af4a5" ["CS2_38"]="98787427c10478425efd4f54d823805724daff1d2fcbd822b3a2d7006a1051d8"
)

total_bytes=0
for entry in "${cells[@]}"; do
    cell="${entry%%|*}"
    kind="${entry#*|}"
    archive="$DEST/_archives/$cell.zip"
    target="$DEST/$cell"
    want="${expected[$cell]:-}"

    if [[ -f "$archive" ]]; then
        got="$(sha256sum "$archive" | cut -d' ' -f1)"
        if [[ -n "$want" && "$got" != "$want" ]]; then
            echo "== $cell: sha256 не совпал, перекачиваю" >&2
            rm -f "$archive"
        else
            echo "== $cell: архив уже есть ($kind)"
        fi
    fi

    if [[ ! -f "$archive" ]]; then
        echo "== $cell: скачиваю ($kind)"
        curl --fail --location --retry 3 --retry-delay 5 --silent --show-error \
            --output "$archive.part" "$BASE/$cell.zip"
        mv "$archive.part" "$archive"
        got="$(sha256sum "$archive" | cut -d' ' -f1)"
        if [[ -n "$want" && "$got" != "$want" ]]; then
            echo "    ОШИБКА sha256: ожидался $want, получен $got" >&2
            exit 1
        fi
        echo "    sha256=$got $( [[ -z "$want" ]] && echo '(не зафиксирован)' )"
    fi

    if [[ -d "$target" && -n "$(ls -A "$target" 2>/dev/null)" ]]; then
        echo "    распаковано ранее, пропускаю"
    else
        mkdir -p "$target"
        unzip -q -o "$archive" -d "$target"
    fi
    size="$(du -sb "$target" | cut -f1)"
    total_bytes=$((total_bytes + size))
    echo "    файлов: $(find "$target" -type f | wc -l), распаковано: $((size / 1024 / 1024)) МиБ"
done

echo
echo "готово: $DEST (всего $((total_bytes / 1024 / 1024)) МиБ распакованных данных)"
echo "дальше: bsb ingest --dataset calce --raw $DEST"
