"""Согласованность упаковки: ``pyproject.toml``, ``uv.lock``, ``.python-version``.

Студент разворачивает окружение одной командой ``uv sync``. Она ставит
dev-инструменты из группы ``dev`` (PEP 735), а не из extras, поэтому списки в
двух местах обязаны совпадать: иначе ``uv sync`` и ``pip install -e ".[dev]"``
дадут разные окружения, и «у меня работает» перестанет что-либо значить.
Проверка дешёвая и ловит расхождение сразу, а не у студента.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from packaging.requirements import Requirement
from packaging.version import Version

REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = REPO_ROOT / "pyproject.toml"
LOCKFILE = REPO_ROOT / "uv.lock"
PYTHON_VERSION = REPO_ROOT / ".python-version"


def _pyproject() -> dict:
    with PYPROJECT.open("rb") as handle:
        return tomllib.load(handle)


def _normalised(requirements: list[str]) -> set[str]:
    """Сравнивать требования надо по имени и спецификатору, а не по строке."""
    parsed = {Requirement(item) for item in requirements}
    return {f"{item.name}{item.specifier}" for item in parsed}


def test_repository_has_pyproject() -> None:
    """Тест бессмыслен без файла, на который он ссылается."""
    assert PYPROJECT.is_file()


def test_dev_group_matches_dev_extra() -> None:
    """Группа ``dev`` для uv и extra ``dev`` для pip перечисляют одно и то же."""
    project = _pyproject()
    extra = project["project"]["optional-dependencies"]["dev"]
    group = project["dependency-groups"]["dev"]

    assert _normalised(group) == _normalised(extra)


def test_dev_group_is_default_for_uv() -> None:
    """``uv sync`` без флагов обязан ставить dev-инструменты.

    Иначе «одна команда» превращается в две, и студент, запустивший просто
    ``uv sync``, не получит ни pytest, ни ruff.
    """
    uv_config = _pyproject().get("tool", {}).get("uv", {})

    assert uv_config.get("default-groups") == ["dev"]


def test_dev_group_declares_what_ci_and_tests_need() -> None:
    """В группе есть pytest, ruff и matplotlib — их требуют тесты и CI."""
    names = {Requirement(item).name for item in _pyproject()["dependency-groups"]["dev"]}

    assert {"pytest", "ruff", "matplotlib"} <= names


def test_lockfile_is_committed() -> None:
    """``uv.lock`` лежит в репозитории: без него версии у студентов разъедутся."""
    assert LOCKFILE.is_file()
    content = LOCKFILE.read_text(encoding="utf-8")

    assert 'name = "battery-soh-bench"' in content


def test_lockfile_requires_python_matches_project() -> None:
    """Ограничение Python в локе и в проекте не разошлось."""
    project = _pyproject()
    locked = tomllib.loads(LOCKFILE.read_text(encoding="utf-8"))

    assert locked["requires-python"] == project["project"]["requires-python"]


def test_pinned_python_satisfies_requires_python() -> None:
    """Интерпретатор из ``.python-version`` удовлетворяет ``requires-python``.

    Если разойдутся, uv попытается поставить окружение на версии, которую сам
    же проект запрещает, и падение будет непонятным.
    """
    project = _pyproject()
    pinned = PYTHON_VERSION.read_text(encoding="utf-8").strip()
    specifier = Requirement(f"python{project['project']['requires-python']}").specifier

    assert specifier.contains(Version(pinned)), (
        f".python-version={pinned} не удовлетворяет requires-python="
        f"{project['project']['requires-python']}"
    )


def test_python_pin_is_committed_not_ignored() -> None:
    """``.python-version`` не должен оказаться в .gitignore.

    Файл был там раньше; если его снова исключат, uv у студентов начнёт
    выбирать интерпретатор «какой найдётся» — ровно то, от чего он страхует.
    """
    ignored = {
        line.strip()
        for line in (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }

    assert ".python-version" not in ignored
    assert PYTHON_VERSION.is_file()
