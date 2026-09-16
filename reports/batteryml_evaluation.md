# BatteryML как источник парсинга сырых данных (SNL/Sandia, CALCE)

## Что проверялось

- Репозиторий склонирован в `.scratch/batteryml`, коммит `2861ae3` (2024-12-18),
  лицензия MIT; файлы тулкита не менялись. Прочитаны `preprocess_SNL.py` (195),
  `preprocess_CALCE.py` (207), `preprocess/base.py`, `preprocess/download.py`,
  `data/battery_data.py`, `label/soh.py`, `label/rul.py`, `__init__.py`,
  `requirements.txt`, `dataprepare.md`, `configs/**/snl.yaml`.
- Дополнительно собран и запущен PyTorch-free срез в throwaway-venv
  (`.scratch/tv`, только numpy/pandas/scipy/numba/tqdm) на синтетических данных:
  Battery Archive–подобные SNL CSV и CADEX `.txt` (скрипты `.scratch/harness/`);
  PyTorch не устанавливался. **Не проверено на реальных данных:** файлов CS2_8/CS2_21
  (CADEX) в репозитории нет и BatteryML их не скачивает; выводы по ним — только из кода.

## SNL/Sandia

- **Раскладка сырья.** Battery Archive, один плоский каталог, по два CSV на ячейку:
  `*_timeseries.csv` и `*_cycle_data.csv`. Ячейка = имя файла без `_timeseries`
  (`preprocess_SNL.py:20-22`); чтение — `pd.read_csv` (`:61-64`); раскладка описана
  в `dataprepare.md:78-95`. Готовых ссылок на скачивание SNL нет (Battery Archive
  требует заявку).
- **Колонки и единицы** (`timeseries`, `:97-107`): `Cycle_Index`, `Voltage (V)`,
  `Current (A)`, `Cell_Temperature (C)`, `Discharge_Capacity (Ah)`,
  `Charge_Capacity (Ah)`, `Test_Time (s)`. Из `cycle_data` — только `Cycle_Index` и
  `Discharge_Capacity (Ah)` (`:159`, `:170`). Единицы из имён колонок, пересчётов нет.
- **Знак тока не интерпретируется:** `Current (A)` перекладывается как есть (`:103`),
  ёмкости берутся из готовых накопленных колонок Battery Archive (`:105-106`), а не
  интегрированием. О знаковом соглашении источника BatteryML ничего не сообщает —
  «заряд +, разряд −» придётся проверять самим. Номер цикла — 1:1 из `Cycle_Index`
  через `groupby` (`:97-100`).
- **Capacity check vs рабочий цикл: не различаются вообще.** В коде нет ни
  `is_reference`, ни детектора опорных циклов. Grep по `capacity.?check|checkup|rpt|SOC`
  по всему репо даёт только захардкоженные `start_soc/end_soc` в протоколах
  (`preprocess_SNL.py:113-119`). Единственная поцикловая фильтрация —
  `clean_snl_cell` (`:146-195`): отбрасывание выбросов по разнице `Qd` с соседями
  (`:159-167`) и импутация пропущенных циклов (`:169-194`). Проверено на синтетике:
  одиночные «проверочные» циклы с `Qd` на 15 % выше рабочих **выживают** как
  обычные циклы, потому что порог `med*10` масштабируется тем же скачком. Разметку
  опорных циклов писать нам.
- **SOH.** `label/soh.py:37-62`: `Qd = max(cycle.discharge_capacity_in_Ah)`,
  `label = Qd / nominal_capacity_in_Ah` (mode `relative`), либо значение из
  `soh_filepath` (json/pkl, `:26-36`). `nominal_capacity_in_Ah` — эвристика по имени
  (`preprocess_SNL.py:82-93`: NMC 3.0/2.7, NCA 3.2/1.92/2.88, иначе LFP 1.1), т.е.
  константа, а не измерение; привязки SOH к capacity check нет. Для SNL в конфигах
  стоит `RULLabelAnnotator` (`configs/baselines/*/snl.yaml`) с порогом
  `nominal * eol_soh` (`label/rul.py:27`).
- **Потери состава.** `to_drop` (`:23-49`) жёстко выбрасывает 25 из 86 ячеек SNL;
  README (строка SNL) подтверждает «Cell Count 61» — копирование дало бы нам на 29 %
  меньше теста.

## CALCE

- **Серии.** Обрабатываются любые положенные пользователем `*.zip` (`:26-27`);
  список загрузки (`download.py:36-49`) — только `CS2_33..CS2_38` и
  `CX2_16, CX2_33..CX2_38`. То есть CS2 и CX2, но неполные: `dataprepare.md:44-60`
  прямо пишет «we did not use all the cells due to data format issues».
- **Excel:** `load_excel` (`:164-193`) читает листы с именем, начинающимся на
  `Channel` (`:175-176`), с fallback на все листы (`:179-181`); оставляет только
  `Cycle_Index, Test_Time(s), Current(A), Voltage(V)` (`:186-187`) и пишет рядом с
  сырьём `*_cache.csv` (`:191`) — побочный эффект в каталоге raw.
- **`.mat` не поддерживается вовсе:** glob только `txt/xlsx/xls` (`:53-56`);
  `loadmat`/`scipy.io` для CALCE нет (grep; `.mat` встречается только у MATR через
  h5py). Заявленная в ТЗ §6.2 «смесь `.mat` и `.xlsx`» BatteryML не покрывает.
- **CADEX `.txt`:** есть общая ветка `load_txt` (`:196-207`) с колонками
  `Charge count`, `Time`, `mA`, `mV`, и регэксп даты `extract_date_from_filename`
  (`:152-161`), совпадающий с именами вида `CS2_8_1_12_2010.txt`. То есть CS2_8/CS2_21
  распарсились бы, если положить zip, но CADEX-специфичного кода и скачивания нет.
  **Риск:** `Cycle_Index = Charge count // 2 + 1` (`:202`) даёт границу цикла между
  разрядом и *следующим* зарядом. Проверено на синтетических полуциклах: цикл 1
  содержит только заряд, последний — только разряд, средние склеивают разряд с
  последующим зарядом. Нулевой по `Qd` первый цикл затем срезается фильтром
  `Qd[i] > 0.1` (`:95`), а хвостовой цикл с нулевым `Qc` остаётся. На реальных
  CADEX-файлах проверить не удалось.
- **Ёмкость/SOH.** `calc_Q` (`:128-138`, `@njit`) интегрирует ток со знаком
  (заряд при `I>0`, разряд при `I<0`) — соглашение о знаке здесь есть. Чистка:
  `medfilt(Qd, 21)` + порог `3*MAD` (`:88-90`), отброс `Qd <= 0.1` (`:95`),
  перенумерация (`:96-98`), ручные правки `CX2_34`/`CX2_16` (`:91-92`, `:103-104`).
  Опорных циклов нет. Метаданные (`:106-115`): `form_factor='prismatic'`,
  `anode='graphite'`, `cathode='LCO'`, `C = 1.1 if 'CS' in cell.upper() else 1.35`
  (`:100`), V 2.7…4.2; протоколы пустые — в коде TODO
  «specify the charge and discharge protocols» (`:99`).

## Связанность с PyTorch

- Любой `import batteryml` тянет torch: `__init__.py:13` импортирует `.models` →
  `models/nn_model.py:5 import torch`; также `__init__.py:4` → `data/__init__.py:4`
  → `data/databundle.py:4 import torch`; `__init__.py:14` (feature) и `:20` (label)
  тоже torch. Оба препроцессора делают `from batteryml import ...`
  (`preprocess_SNL.py:11`, `preprocess_CALCE.py:17`), а `preprocess/base.py:9` —
  `from batteryml import BatteryData`, поэтому и SNL-файл не импортируется без torch.
  Проверено запуском: в venv с pandas, но без torch, `import batteryml` падает на
  `databundle.py:4` с `ModuleNotFoundError: No module named 'torch'`.
- При этом torch **не объявлен** в `requirements.txt` (14 строк, torch отсутствует),
  а numpy запинен `<2.0.0` (`requirements.txt:4`) — конфликт с numpy 2.5.3 тулкита.
- **Размер PyTorch-free ядра:** `data/battery_data.py` 168 + `builders.py` 11 +
  `utils/registry.py` 60 + `preprocess/base.py` 53 + `preprocess_SNL.py` 195 +
  `preprocess_CALCE.py` 207 = **694 строки**. Я собрал их в `.scratch/harness`,
  заменив только `__init__.py` (1–2 строки, логику не трогал), и оба препроцессора
  отработали на numpy/pandas/scipy/tqdm/numba. Вывод: парсинг **отделим, но только
  копированием файлов**; как установленный пакет — нет. Дополнительные зависимости
  сверх §2: pandas, numba (`preprocess_CALCE.py:12`), openpyxl, tqdm, `scipy.signal`.

## Метаданные

- Поля `BatteryData` (`data/battery_data.py:80-127`). Для SNL заполняются
  (`preprocess_SNL.py:131-143`): `cell_id`, `form_factor='cylindrical_18650'`,
  `anode_material`, `cathode_material` из имени, эвристический
  `nominal_capacity_in_Ah`, min/max V и max ток по катоду (`:124-127`), скорости
  заряда/разряда из имени (`:110-120`).
- Против схемы тулкита `cells` (`src/bsb/schema.py:107-118`) не хватает
  `temperature_setpoint_c` (в имени есть `25C/15C/35C`, но в поле не парсится),
  `batch_id` и главного — `bol_capacity_ah`: измеренной ёмкости первого цикла нет,
  только константа. `charge_protocol` для CALCE пуст; `chemistry` — только катод
  (CALCE всё `LCO`, хотя CS2 1.1 А·ч и CX2 1.35 А·ч различаются). У SNL не
  заполняется и `depth_of_discharge`: дефолт `1.0` (`battery_data.py:91`) даже для
  `20-80`/`40-60`, а `charge_protocol` жёстко `start_soc=0.0, end_soc=1.0`
  (`:111-115`) — для частичного DOD неверно.

## Вердикт по SNL

**Borrow logic + write our own; как зависимость — нет.** Переиспользовать стоит
ровно контракт сырья: какие два CSV на ячейку, какие имена колонок, как из имени
ячейки достаются химия, температура, DOD и скорости (`preprocess_SNL.py:61-64,
96-108, 110-127`) — это экономит ~30–40 строк маппинга. Всё остальное, что нам
нужно, BatteryML не даёт: детекции capacity check нет, `step_type` (CC/CV/rest) нет
вовсе — только целые циклы, измеренного `bol_capacity_ah` нет, температура/DOD/batch
не парсятся, 25 из 86 ячеек выброшены. Поэтому §5/§4 (опорные циклы, `is_reference`,
`soh_source`) придётся писать с нуля в любом случае; переиспользование не сокращает
критический путь, а `clean_snl_cell` (`:146-195`) с forward-импутацией прямо
противоречит запрету §4.2 на фиктивные измеренные циклы.

## Вердикт по CALCE

**Write our own, позаимствовав две вещи.** (1) `load_txt` (`:196-207`) вместе с
регэкспом даты (`:152-161`) — единственный в BatteryML код, который вообще трогает
CADEX; перед доверием `Charge count // 2 + 1` обязательно сверить спаривание
полуциклов на реальных CS2_8/CS2_21. (2) `load_excel` (`:164-193`) подсказывает,
какие листы Arbin читать (`Channel*`) и какие имена колонок ожидать
(`Cycle_Index, Test_Time(s), Current(A), Voltage(V)`). Как зависимость не годится:
`.mat` не читается, покрытие CS2/CX2 неполное, `numba`+`pandas`+`openpyxl` вне §2,
numpy-пин `<2.0` конфликтует. Ничего не теряем: наш §17.1 даёт единый CC-CV 0.5C для
CS2, так что `charge_protocol` берётся из ТЗ, а не из BatteryML.

## Что именно стоит позаимствовать

1. SNL: контракт Battery Archive-файлов и разбор имени ячейки — `preprocess_SNL.py:61-64, 96-108, 110-127`. Копировать подход, не код (pandas → polars; `step_type` нам всё равно нужен свой).
2. SNL: идея `clean_snl_cell` (`:146-195`) как детектор глитчей и пропущенных циклов — но импутацию заменить на отбрасывание (§4.2 запрещает фабриковать измеренные циклы).
3. CALCE: `load_txt` + `extract_date_from_filename` (`:196-207`, `:152-161`) — единственный пример разбора CADEX `.txt`; сначала проверить `Charge count // 2 + 1`.
4. CALCE: `load_excel` (`:164-193`) — выбор листов `Channel*` и набор колонок Arbin.
5. `label/soh.py:37-62`: формула `relative` SOH и хук `soh_filepath` (json/pkl) концептуально совпадают с §5, но номинал надо брать измеренный, а не эвристику `get_capacity`. `builders.py` + `utils/registry.py` — образец pluggable-реестра, но для четырёх адаптеров он, скорее всего, избыточен.

**Главный отрицательный вывод:** BatteryML — это RUL-препроцессинг целых циклов, а не SOH-бенчмарк по опорным циклам: понятий «capacity check / reference cycle» и `step_type` в нём нет ни для SNL, ни для CALCE.
