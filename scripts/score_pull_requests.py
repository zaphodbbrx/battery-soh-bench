"""Доверенная оценка CSV в открытых PR без запуска кода участников.

Этот скрипт запускается только из базовой ветки workflow pull_request_target.
Из публичного head-репозитория скачивается только submission.csv как недоверенные данные.
"""

from __future__ import annotations

import base64
import gzip
import io
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

import polars as pl

from bsb.config import load_config
from bsb.eval import evaluate
from bsb.validate_submission import coerce_numeric_columns, expected_test_ids, validate_submission

API = "https://api.github.com"
USER_AGENT = "battery-soh-bench-evaluator"
LEADERBOARD_TITLE = "Лидерборд battery-soh-bench"
LEADERBOARD_MARKER = "<!-- bsb-leaderboard -->"
SCORE_MARKER = "<!-- bsb-score -->"
SUBMISSION_PATTERN = re.compile(r"submissions/([a-z0-9][a-z0-9-]{0,49})/submission\.csv\Z")
MAX_SUBMISSION_BYTES = 250_000
SOURCE_REPOSITORY_PATTERN = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+\Z")
COMMIT_SHA_PATTERN = re.compile(r"[0-9a-f]{40}\Z")


def _request(
    url: str, *, token: str, method: str = "GET", payload: dict[str, Any] | None = None
) -> tuple[bytes, Any]:
    """Выполнить запрос к GitHub API, не печатая тело ошибки или токен."""
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "User-Agent": USER_AGENT,
        "X-GitHub-Api-Version": "2022-11-28",
    }
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    if data is not None:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        response = urllib.request.urlopen(request, timeout=30)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"GitHub API вернул HTTP {exc.code} для {method} {url}") from None
    with response:
        return response.read(), response.headers


def _api_json(
    path: str, *, token: str, method: str = "GET", payload: dict[str, Any] | None = None
) -> Any:
    body, _ = _request(f"{API}{path}", token=token, method=method, payload=payload)
    return json.loads(body) if body else None


def _api_pages(path: str, *, token: str) -> list[dict[str, Any]]:
    """Собрать все страницы list API."""
    url = f"{API}{path}"
    result: list[dict[str, Any]] = []
    while url:
        body, headers = _request(url, token=token)
        result.extend(json.loads(body))
        next_link = re.search(r'<([^>]+)>;\s*rel="next"', headers.get("Link", ""))
        url = next_link.group(1) if next_link else ""
    return result


def _download_submission(pr: dict[str, Any], filename: str) -> bytes:
    """Загрузить только CSV из публичного head-репозитория, без API-токена."""
    head = pr.get("head") or {}
    head_repository = head.get("repo") or {}
    source_repo = head_repository.get("full_name", "")
    sha = head.get("sha", "")
    if not SOURCE_REPOSITORY_PATTERN.fullmatch(source_repo) or not COMMIT_SHA_PATTERN.fullmatch(
        sha
    ):
        raise ValueError("неподдерживаемый head репозиторий или SHA")
    path = urllib.parse.quote(filename, safe="/")
    url = f"https://raw.githubusercontent.com/{source_repo}/{sha}/{path}"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        response = urllib.request.urlopen(request, timeout=30)
    except urllib.error.HTTPError as exc:
        raise FileNotFoundError(f"файл недоступен (HTTP {exc.code})") from None
    with response:
        size = response.headers.get("Content-Length")
        if size and int(size) > MAX_SUBMISSION_BYTES:
            raise ValueError("CSV больше 250 КБ")
        content = response.read(MAX_SUBMISSION_BYTES + 1)
    if len(content) > MAX_SUBMISSION_BYTES:
        raise ValueError("CSV больше 250 КБ")
    return content


def _truth_from_secret(value: str) -> pl.DataFrame:
    """Декодировать сжатый CSV из Actions Secret в памяти."""
    packed = base64.b85decode(value.encode("ascii"))
    csv_bytes = gzip.decompress(packed)
    return pl.read_csv(io.BytesIO(csv_bytes), schema_overrides={"fragment_id": pl.String})


def _pr_submission(
    repo: str,
    pr: dict[str, Any],
    *,
    token: str,
    config: Any,
    expected_ids: list[str],
    truth: pl.DataFrame,
) -> dict[str, Any] | None:
    """Оценить CSV одного PR; PR без файла сабмита пропустить."""
    number = int(pr["number"])
    pr_url = pr["html_url"]
    files = _api_pages(f"/repos/{repo}/pulls/{number}/files?per_page=100", token=token)
    candidates = [
        (match.group(1), item["filename"])
        for item in files
        if item.get("status") != "removed"
        and (match := SUBMISSION_PATTERN.fullmatch(item["filename"]))
    ]
    if not candidates:
        return None
    if len(candidates) != 1:
        return {
            "pr": number,
            "slug": f"pr-{number}",
            "url": pr_url,
            "status": "нужен один submission.csv",
        }

    slug, filename = candidates[0]
    try:
        csv_bytes = _download_submission(pr, filename)
    except FileNotFoundError:
        return {"pr": number, "slug": slug, "url": pr_url, "status": "CSV недоступен в head PR"}
    except ValueError as exc:
        return {"pr": number, "slug": slug, "url": pr_url, "status": str(exc)}

    try:
        submission = pl.read_csv(io.BytesIO(csv_bytes), schema_overrides={"fragment_id": pl.String})
        submission, _ = coerce_numeric_columns(submission)
        report = validate_submission(submission, expected_ids=expected_ids, config=config)
    except Exception:
        return {
            "pr": number,
            "slug": slug,
            "url": pr_url,
            "status": "CSV не удалось прочитать",
        }
    if not report.ok:
        return {
            "pr": number,
            "slug": slug,
            "url": pr_url,
            "status": f"ошибки валидации: {len(report.errors)}",
        }
    allowed_columns = [
        column
        for column in ("fragment_id", "soh_pred", "soh_lo", "soh_hi")
        if column in submission.columns
    ]
    submission = submission.select(allowed_columns)

    main = evaluate(submission, truth, config=config)
    hard = evaluate(submission, truth, config=config, hard_mode=True)
    return {
        "pr": number,
        "slug": slug,
        "url": pr["html_url"],
        "status": "ok",
        "main": main["overall"],
        "hard": hard["overall"],
    }


def _score_text(result: dict[str, Any]) -> str:
    if result["status"] != "ok":
        return f"Сабмит не оценён: {result['status']}. Исправьте PR и отправьте обновление."
    main = result["main"]
    hard = result["hard"]
    return (
        f"**Основная номинация (NCA + NMC):** MAE {main['mae_pp']:.4f} п.п.; "
        f"P95 {main['p95_abs_pp']:.4f} п.п. (n={main['n']}).\n\n"
        f"**LFP:** MAE {hard['mae_pp']:.4f} п.п.; "
        f"P95 {hard['p95_abs_pp']:.4f} п.п. (n={hard['n']})."
    )


def _upsert_comment(repo: str, issue_number: int, marker: str, body: str, *, token: str) -> None:
    """Создать или обновить только комментарий, помеченный этим evaluator."""
    comments = _api_pages(f"/repos/{repo}/issues/{issue_number}/comments?per_page=100", token=token)
    existing = next(
        (
            item
            for item in comments
            if marker in item.get("body", "")
            and item.get("user", {}).get("login") == "github-actions[bot]"
        ),
        None,
    )
    payload = {"body": f"{marker}\n{body}"}
    if existing:
        _api_json(
            f"/repos/{repo}/issues/comments/{existing['id']}",
            token=token,
            method="PATCH",
            payload=payload,
        )
    else:
        _api_json(
            f"/repos/{repo}/issues/{issue_number}/comments",
            token=token,
            method="POST",
            payload=payload,
        )


def _leaderboard_issue(repo: str, *, token: str) -> int:
    issues = _api_pages(f"/repos/{repo}/issues?state=open&per_page=100", token=token)
    existing = next(
        (
            item
            for item in issues
            if item["title"] == LEADERBOARD_TITLE and "pull_request" not in item
        ),
        None,
    )
    if existing:
        return int(existing["number"])
    issue = _api_json(
        f"/repos/{repo}/issues",
        token=token,
        method="POST",
        payload={
            "title": LEADERBOARD_TITLE,
            "body": (
                "Автоматический лидерборд. Последние оценки открытых PR "
                "обновляются при каждом изменении сабмита."
            ),
        },
    )
    return int(issue["number"])


def _leaderboard_body(results: list[dict[str, Any]]) -> str:
    valid = sorted(
        (item for item in results if item["status"] == "ok"),
        key=lambda item: (item["main"]["mae_pp"], item["pr"]),
    )
    invalid = sorted(
        (item for item in results if item["status"] != "ok"), key=lambda item: item["pr"]
    )
    lines = [
        "## Текущие результаты",
        "",
        "Открытые PR; ранжирование по MAE основной номинации, меньше — лучше.",
        "Код участников не запускается. Оценщик читает только `submission.csv`.",
        "",
        "| Место | Команда | PR | MAE, п.п. | P95, п.п. | LFP MAE, п.п. |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for place, item in enumerate(valid, start=1):
        lines.append(
            f"| {place} | `{item['slug']}` | [#{item['pr']}]({item['url']}) | "
            f"{item['main']['mae_pp']:.4f} | {item['main']['p95_abs_pp']:.4f} | "
            f"{item['hard']['mae_pp']:.4f} |"
        )
    if not valid:
        lines.append("| — | Пока нет корректных сабмитов | — | — | — | — |")
    if invalid:
        lines.extend(["", "### PR без оценки", ""])
        lines.extend(
            f"- `{item['slug']}` ([#{item['pr']}]({item.get('url', '')})): {item['status']}"
            for item in invalid
        )
    return "\n".join(lines)


def main() -> int:
    """Оценить все открытые PR целиком и обновить PR-комментарии и issue."""
    repo = os.environ["GITHUB_REPOSITORY"]
    token = os.environ["GITHUB_TOKEN"]
    config = load_config()
    secret = os.environ.get("BSB_TEST_TRUTH_B85", "")
    if not secret:
        raise SystemExit("Не задан GitHub Actions Secret BSB_TEST_TRUTH_B85")
    truth = _truth_from_secret(secret)
    expected_ids = expected_test_ids(config)
    if expected_ids is None:
        raise SystemExit("В доверенной базовой ветке отсутствует test_unlabeled.parquet")
    open_prs = _api_pages(f"/repos/{repo}/pulls?state=open&base=main&per_page=100", token=token)
    results = [
        result
        for pr in open_prs
        if (
            result := _pr_submission(
                repo,
                pr,
                token=token,
                config=config,
                expected_ids=expected_ids,
                truth=truth,
            )
        )
        is not None
    ]

    for result in results:
        if "url" in result:
            _upsert_comment(
                repo,
                result["pr"],
                SCORE_MARKER,
                _score_text(result),
                token=token,
            )
    issue = _leaderboard_issue(repo, token=token)
    _upsert_comment(
        repo,
        issue,
        LEADERBOARD_MARKER,
        _leaderboard_body(results),
        token=token,
    )

    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as stream:
            stream.write(_leaderboard_body(results) + "\n")
    print(f"Оценено открытых PR: {len(results)}; лидерборд обновлён.")

    if os.environ.get("PR_ACTION") != "closed":
        current_pr = int(os.environ["PR_NUMBER"])
        current_result = next((item for item in results if item["pr"] == current_pr), None)
        if current_result is None or current_result["status"] != "ok":
            print(f"Текущий PR не прошёл оценку: {current_result}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Ошибка оценщика: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
