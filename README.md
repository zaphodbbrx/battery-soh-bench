# battery-soh-bench

Тулкит студенческого соревнования: по случайному фрагменту зарядной кривой
фиксированной ширины по напряжению предсказать нормированную остаточную
ёмкость (SOH) литий-ионной ячейки на момент этого цикла.

Обучение — CALCE, KIT NMC/C-SiO, Oxford. Скрытый тест — Sandia (NCA + NMC;
LFP — отдельная бонусная номинация).

Техническое задание — [battery-soh-bench-spec.md](battery-soh-bench-spec.md).
Он же единственный источник правды по решениям; ссылки вида «§5.1» ниже —
на его разделы.

## Установка

```bash
git clone <repo> && cd battery-soh-bench
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
bsb config-check
```

Данные в репозиторий не входят (ТЗ §3). Скачайте их скриптами из `scripts/`,
затем выполните `scripts/prepare_all.sh`.

## Быстрый старт

```bash
bsb config-check                          # проверить конфиг
bsb ingest --dataset oxford --raw data/raw/oxford
bsb compact                               # собрать унифицированный уровень
bsb splits-check                          # утечка ячеек между сплитами
bsb fragments --split val --mode fixed
bsb baseline --model elastic_net
bsb eval --submission submission.csv
```

## Раскладка

```
configs/benchmark.yaml   единственный источник правды по параметрам (ТЗ §3)
splits/                  списки cell_id по сплитам, коммитятся (ТЗ §8)
data/                    gitignore; наполняется скриптами
src/bsb/
  config.py              строгий загрузчик конфига
  schema.py              схема §4, валидаторы, CellBundle
  store.py               раскладка унифицированного уровня (parts -> compact)
  splits.py              сплиты и assert_no_cell_leakage
  ingest/                адаптеры датасетов (§6)
  labels.py              опорные циклы и soh (§5)
  fragments.py           генератор фрагментов (§7)
  dataset.py features.py baseline.py eval.py validate_submission.py
  cli.py                 команда bsb (§13)
tests/                   round-trip, детерминизм, утечки, валидаторы (§14)
```

## Решения по открытым вопросам §18

| № | Вопрос | Решение |
|---|---|---|
| 1 | Только опорные циклы или интерполяция (§5.1) | **Только опорные**; интерполяция — опциональный флаг `labels.allow_interpolated` |
| 2 | LFP: бонус или обучение (§9) | **Бонусная номинация**; основной тест NCA + NMC, LFP считается с `--hard-mode` |
| 3 | Пороги CC/CV (§4.4) | Вынесены в `step_labeling` конфига |
| 4 | `soh_gap_cycles` (§5) | `labels.max_soh_gap_cycles = 60`; в режиме «только опорные» не действует |
| 5 | Серии CALCE (§6.2) | `ingest.calce.series = [CS2]` — **открыто** |
| 6 | Прореживание KIT (§6.3) | `ingest.kit.target_resolution_s = 10` — **открыто** |
| 7 | Календарное старение KIT (§6.3) | `include_calendar_aging = false` — **открыто** |
| 8 | Сетка `delta_v` (§10) | `[0.05, 0.1, 0.15, 0.2]` — **открыто** |
| 9 | BatteryML как основа адаптеров (§17.5) | Свои адаптеры — **открыто** |
| — | Подшаг заряда Oxford: `OCVch` (C/18.5) или `C1ch` (1C) | `OCVch` по §17.3, меняется в `ingest.oxford.charge_step` |
| — | Неполные характеризации Oxford (2 из 519) | Помечены в `notes`; отбрасывать или нет — решает владелец |
| 10 | Порог пробы на утечку (§12.3) | `max_relative_improvement = 0.15` — **открыто** |

## Данные

```bash
bash scripts/download_oxford.sh          # ORA, 253.8 МБ, sha256 проверяются
bsb ingest --dataset oxford --raw data/raw/oxford --compact
python scripts/acceptance_plots.py --dataset oxford --cells 5
```

`scripts/prepare_all.sh` прогоняет ingest → compact → fragments по всем
разложенным датасетам.

## Статус

| фаза §15 | что | состояние |
|---|---|---|
| 0 | схема §4, `benchmark.yaml`, каркас, тесты, CI | закрыта |
| 1 | адаптер Oxford end-to-end + приёмка по графикам | закрыта, ждёт вашего просмотра `reports/acceptance_oxford.png` |
| 2 | `labels.py`, правила опорных циклов | арифметика §5 готова; правила по Sandia и KIT — за вами |
| 3 | `fragments.py` | не начата |
| 4–6 | адаптеры CALCE, KIT, Sandia | не начаты |
| 7–8b | сплиты, `eval.py`, валидатор, бейзлайны | не начаты |

Команды, ещё не реализованные, возвращают код 3 и называют свою фазу: тулкит не
делает вид, что что-то посчитал.

### Уточнение к §6 ТЗ

ТЗ §6 предполагает, что `load_cell` сразу возвращает `CellBundle` по схеме §4.
Это невозможно без подлога: `cells.bol_capacity_ah` и `cycles.soh` выводятся из
опорных циклов, а какие циклы опорные — определяется правилом, которое
применяется следующим шагом. Поэтому адаптеры возвращают промежуточный
`RawBundle` (ёмкость и timeseries как в исходнике) и уже из него
`mark_reference_cycles` через `bsb.labels.build_labels` собирает §4. Так ни одно
производное поле не заполняется «чем-нибудь похожим».

## Лицензии данных

Каждый набор цитируется по требованию правообладателя: Oxford — ODbL 1.0 /
DbCL 1.0 плюс диссертация C. R. Birkl; Sandia — Battery Archive и
Preger et al.; KIT — Luh & Blank, *Scientific Data*; CALCE — University of
Maryland. Подробности — §17 ТЗ.
