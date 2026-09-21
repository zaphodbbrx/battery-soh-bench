# battery-soh-bench

Тулкит студенческого соревнования: по фрагменту зарядной кривой литий-ионной
ячейки предсказать **SOH** — насколько ячейка изношена.

Никакой химии для участия не нужно: [ликбез](docs/01-ликбез.md) объясняет всё
на пальцах за десять минут.

## Что от вас хотят

На вход даётся кусок зарядной кривой — окно шириной 0.2 В, 128 точек, три
канала (заряд, ток, температура). Нужно вернуть одно число `soh_pred` от 0 до 1:
долю сохранившейся ёмкости на момент этого цикла. Сдаётся CSV с
`fragment_id,soh_pred`.

Номера цикла, истории ячейки и её идентификатора в данных нет — только сам
фрагмент. Обучение и тест — **разные наборы данных**, так что запомнить ячейки
не получится.

## Документация

| документ | о чём |
|---|---|
| [Ликбез](docs/01-ликбез.md) | что такое аккумулятор, износ, SOH и зарядная кривая |
| [Задача и данные](docs/02-задача-и-данные.md) | что на входе, что предсказать, откуда брать данные |
| [Сабмит и метрики](docs/03-сабмит-и-метрики.md) | формат ответа, валидатор, номинации, как считаются баллы |
| [Ориентиры](docs/04-ориентиры.md) | бейзлайны и потолок: какой результат хороший |
| [Заметки разработчика](docs/dev/заметки-разработчика.md) | устройство тулкита и решения по спорным местам — организаторам |

Техническое задание целиком — [battery-soh-bench-spec.md](battery-soh-bench-spec.md).
Ссылки вида «§5» в документации указывают на его разделы.

## Установка

Нужен только [uv](https://docs.astral.sh/uv/) — он сам скачает нужный Python
(3.12), создаст окружение и поставит зависимости строго по `uv.lock`:

```bash
git clone https://github.com/zaphodbbrx/battery-soh-bench.git
cd battery-soh-bench
uv sync
uv run bsb config-check
```

Одна команда `uv sync` делает всё: интерпретатор, `.venv`, сам пакет в режиме
editable и dev-инструменты (`pytest`, `ruff`). Дальше любую команду запускайте
через `uv run` — например `uv run pytest` или `uv run bsb baseline`; активировать
окружение вручную не нужно.

Установка uv — одна строка: `curl -LsSf https://astral.sh/uv/install.sh | sh`
(или `pipx install uv`, или `brew install uv`).

<details>
<summary>Если uv недоступен: путь через pip</summary>

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
bsb config-check
```

Список dev-зависимостей живёт в двух местах — `[project.optional-dependencies]`
для pip и `[dependency-groups]` для uv; совпадение проверяет
`tests/test_packaging.py`, так что расходиться они не могут.

</details>

## Быстрый старт

```bash
uv run bsb config-check                   # проверить конфиг
bash scripts/prepare_all.sh               # скачать и разобрать данные (см. ниже)
uv run bsb baseline --model elastic_net   # посчитать бейзлайн и получить сабмит
uv run bsb validate --submission submission.csv
uv run bsb eval --submission submission.csv
```

Полезные команды:

| команда | что делает |
|---|---|
| `bsb baseline --model NAME` | бейзлайн и готовый сабмит; NAME — `constant`, `probe` или `elastic_net` |
| `bsb noise-floor` | шумовой пол данных — «потолок» точности |
| `bsb eval --submission FILE [--hard-mode]` | посчитать метрики (LFP — с `--hard-mode`) |
| `bsb validate --submission FILE` | проверить формат сабмита |
| `bsb labels --dataset X` | переприменить правила разметки, не перечитывая сырые файлы |
| `bsb splits-check` | проверить, что ячейки не пересекаются между сплитами |

## Данные

Сырые наборы в репозиторий не входят — это гигабайты. Скрипты скачивания лежат
в `scripts/` и проверяют sha256:

```bash
bash scripts/download_calce.sh        # обучение, ~1.35 ГБ
bash scripts/download_oxford.sh       # обучение, ~254 МБ
bash scripts/download_sandia_rd.sh    # обучение, ~312 МБ
bash scripts/download_sandia.sh       # тест, ~115 МБ
bash scripts/prepare_all.sh           # ingest -> compact -> фрагменты
```

Что где лежит и как устроен вход — [docs/02-задача-и-данные.md](docs/02-задача-и-данные.md).

## Раскладка

```
pyproject.toml           зависимости и инструменты; dev-список общий для uv и pip
uv.lock                  зафиксированные версии, коммитится
.python-version          версия интерпретатора для uv (3.12), коммитится
configs/benchmark.yaml   единственный источник правды по параметрам (ТЗ §3)
splits/                  списки cell_id по сплитам, коммитятся (ТЗ §8)
data/                    gitignore; наполняется скриптами
docs/                    документация: ликбез, задача, сабмит, ориентиры
examples/                пример сабмита
src/bsb/
  config.py              строгий загрузчик конфига
  schema.py              схема §4, валидаторы, CellBundle
  store.py               раскладка унифицированного уровня (parts -> compact)
  ingest/                адаптеры наборов: oxford, calce, sandia, sandia_rd
  labels.py              правила опорных циклов и расчёт SOH (§5)
  fragments.py           генератор фрагментов (§7)
  dataset.py features.py baseline.py eval.py validate_submission.py
  cli.py                 команда bsb (§13)
tests/                   round-trip, детерминизм, утечки, валидаторы (§14)
scripts/                 скачивание данных, подготовка, фикстуры, графики приёмки
```

## Лицензии данных

Каждый набор цитируется по требованию правообладателя: Oxford — ODbL 1.0 /
DbCL 1.0 плюс диссертация C. R. Birkl; Sandia — Battery Archive и Preger et al.;
KIT — Luh & Blank, *Scientific Data*; CALCE — University of Maryland.
Подробности — §17 ТЗ.

Код тулкита — под лицензией [MIT](LICENSE).
