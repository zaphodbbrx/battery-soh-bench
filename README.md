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
bsb fragments --split val --mode fixed        # артефакт val/test (после фазы 7)
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
| — | Неполные характеризации Oxford (2 из 519) | Решено: `require_full_charge` исключает их из опорных; 517 измерений |
| — | Подшаг заряда Oxford: `OCVch` (C/18.5) или `C1ch` (1C) | `OCVch` по §17.3, меняется в `ingest.oxford.charge_step` |
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
| 3 | `fragments.py` | закрыта: `fixed` + детерминизм + `sampling` через `bsb.dataset` |
| 4 | адаптер CALCE | разбор закрыт (14/14 ячеек), **таргет ждёт вашего решения** по типу 3 |
| 5–6 | адаптеры KIT, Sandia | не начаты |
| 6 | адаптер Sandia | **заблокирована данными**: у Battery Archive нет программного доступа (см. ниже) |
| 7 | сплиты, `eval.py`, валидатор | `eval.py` и валидатор закрыты; генерация сплитов ждёт Sandia |
| 8–8b | бейзлайны | не начаты |

Команды, ещё не реализованные, возвращают код 3 и называют свою фазу: тулкит не
делает вид, что что-то посчитал.

### Доступ к данным: где мы упёрлись

| датасет | состояние |
|---|---|
| Oxford | скачивается скриптом, sha256 зафиксированы, разобран целиком |
| CALCE | скачивается скриптом (15 архивов, 1.35 ГБ, sha256 зафиксированы), разобран целиком |
| Sandia | **нет программного доступа.** Battery Archive — это JS-фронтенд над Redash, анонимный API закрыт (`/api/dashboards/...` → 404), данные лежат в Postgres (`battery-archive-sandbox/datasources/postgres/cell_data/init.sql`), а FAQ на вопрос «как скачать много ячеек сразу» отвечает «напишите на info@batteryarchive.org» |
| KIT | DOI из §17.2 (`10.35097/kww7jv8ajuvchcah`) на Zenodo содержит только Readme.txt на 165 байт; сами логи лежат в RADAR4KIT |

Оценка BatteryML (§17.5, §18 п.9) — в `reports/batteryml_evaluation.md`:
переиспользовать как зависимость нельзя (импорт тянет PyTorch, что запрещено §2,
плюс numpy<2), а как источник логики он бесполезен для главного — в нём **нет
понятия опорного цикла** и нет `step_type`. Свои адаптеры; позаимствовать стоит
только чтение CADEX-`.txt` для CS2_8/CS2_21.

### Открытый вопрос по таргету CALCE (тип 3)

У CS2 тип 3 (`CS2_3`, `CS2_9`) меняет ток разряда каждый цикл: 0.11, 0.22,
0.55, 1.1, 1.65, 2.2 А. Ёмкость зависит от тока — **1.098 А·ч при 0.11 А против
0.830 А·ч при 2.2 А**, — а `soh` считается от разрядной ёмкости. Соседние циклы
поэтому скачут на 25% по режиму, а не по старению, и кривая SOH (график §14)
перестаёт быть монотонной: `reports/acceptance_calce.png`.

Механизм фильтра написан (`labels.reference_cycle_rules.calce.
require_modal_discharge_current`), но по умолчанию **выключен**: наивное «оставить
модальный ток» ломается на типах 5-6 (полные циклы редки и идут другим током —
ячейка остаётся без `bol`) и на типе 3 сдвигает первый опорный цикл, давая
`soh > 1.2`. Нужно решение владельца (§5): исключить тип 3, считать `bol` по
отфильтрованным циклам или нормировать ёмкость на ток разряда.

Тот же вопрос стоит и между датасетами: Oxford разряжается на C/18.5, CALCE —
на 0.5C или 1C, Sandia — на 0.5C. Это систематический сдвиг в **таргете**, а не
во входе.

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
