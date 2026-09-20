#!/usr/bin/env bash
# Скачивание Sandia (SNL) — скрытого теста бенчмарка (ТЗ §8, §17.4).
#
# Источник: BatteryLife (KDD 2025), открытый сводный датасет на Zenodo,
# лицензия CC-BY-4.0. Внутри — подмножество SNL в уже стандартизованном виде:
# на цикл лежат current/voltage/charge_capacity/discharge_capacity/time/
# temperature, то есть ровно то, что нужно §4.2-§4.3.
#
# Почему не batteryarchive.org напрямую. На Battery Archive данные лежат в
# Postgres за Redash: программного доступа нет (анонимный API отвечает
# «Please login and try again»), а массовая выгрузка, по их же FAQ, только по
# письму. BatteryLife даёт тот же SNL открыто и воспроизводимо, поэтому
# источник заменён осознанно (см. README, «Источник теста»).
#
# Состав: 61 ячейка — NCA 22, NMC 21, LFP 18; температуры 15/25/35 °C;
# заряд у всех 0.5C (это и делает набор пригодным для фрагментов, §5.1).
# В исходном исследовании 86 ячеек; BatteryLife публикует те, для которых есть
# метки ресурса, поэтому группа 40-60 % SOC в подмножество не попала.
#
# Атрибуция обязательна тройная:
#   BatteryLife  — https://doi.org/10.1145/3711896.3737372
#   Battery Archive — https://www.batteryarchive.org/
#   Preger et al., "Degradation of Commercial Lithium-ion Cells as a Function
#   of Chemistry and Cycling Conditions", J. Electrochem. Soc. 167 (2020).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${1:-$ROOT/data/raw/sandia}"
ARCHIVE_URL="https://zenodo.org/api/records/19688272/files/SNL.zip/content"
ARCHIVE="$DEST/_archives/SNL.zip"
#: sha256 архива SNL.zip; зафиксирован после первой загрузки.
EXPECTED_SHA256="1ce27dc94d5d3a9a10ed875805a880c9c0701833915f1b78b09a759c72cfd57a"

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
    echo "== SNL.zip: уже есть, проверяю"
    verify || { echo "    перекачиваю" >&2; rm -f "$ARCHIVE"; }
fi

if [[ ! -f "$ARCHIVE" ]]; then
    echo "== SNL.zip: скачиваю (BatteryLife, Zenodo, CC-BY-4.0)"
    curl --fail --location --retry 3 --retry-delay 5 --silent --show-error \
        --output "$ARCHIVE.part" "$ARCHIVE_URL"
    mv "$ARCHIVE.part" "$ARCHIVE"
    verify
fi

count_before="$(find "$DEST" -maxdepth 2 -name '*.pkl' | wc -l)"
if [[ "$count_before" -eq 0 ]]; then
    echo "== распаковка"
    unzip -q -o "$ARCHIVE" -d "$DEST"
else
    echo "== распаковано ранее ($count_before файлов), пропускаю"
fi

count="$(find "$DEST" -name '*.pkl' | wc -l)"
size="$(du -sh --exclude=_archives "$DEST" | cut -f1)"
echo
echo "готово: $DEST"
echo "ячеек (.pkl): $count, объём: $size"
echo "дальше: bsb ingest --dataset sandia --raw $DEST"
