"""Тесты метрик и разбивок (ТЗ §10)."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest

from bsb.config import BenchmarkConfig
from bsb.eval import evaluate, load_truth, render_report, run_eval, save_results


def _truth(n_chemistry: int = 6, n_temp: int = 3) -> pl.DataFrame:
    """Синтетический эталон теста: NCA, NMC и LFP на разных температурах."""
    chemistries = ["NCA", "NMC", "LFP"]
    temperatures = [15.0, 25.0, 35.0][:n_temp]
    rows: list[dict[str, Any]] = []
    index = 0
    for chemistry in chemistries:
        for temperature in temperatures:
            for step in range(n_chemistry // 3 + 1):
                rows.append(
                    {
                        "fragment_id": f"f{index}",
                        "cell_id": f"sandia_{chemistry}_{temperature}_{step}",
                        "cycle_index": 1 + 100 * step,
                        "soh": round(1.0 - 0.02 * index, 6),
                        "chemistry": chemistry,
                        "temperature_setpoint_c": temperature,
                        "dataset": "sandia",
                    }
                )
                index += 1
    return pl.DataFrame(rows)


def _submission_for(
    truth: pl.DataFrame, *, error: float = 0.0, width: float | None = None
) -> pl.DataFrame:
    """Сабмит с постоянной ошибкой и, при желании, с интервалами."""
    frame = truth.select("fragment_id", "soh").with_columns(
        (pl.col("soh") + error).alias("soh_pred")
    )
    if width is not None:
        frame = frame.with_columns(
            (pl.col("soh_pred") - width).alias("soh_lo"),
            (pl.col("soh_pred") + width).alias("soh_hi"),
        )
    return frame.drop("soh")


def test_perfect_predictions_give_zero_error(repo_config: BenchmarkConfig) -> None:
    """Идеальный сабмит: MAE и P95 равны нулю, покрытие полное."""
    truth = _truth()
    result = evaluate(_submission_for(truth), truth, config=repo_config)
    assert result["overall"]["mae_pp"] == pytest.approx(0.0)
    assert result["overall"]["p95_abs_pp"] == pytest.approx(0.0)
    assert result["coverage_of_test"] == pytest.approx(1.0)


def test_mae_is_in_percentage_points(repo_config: BenchmarkConfig) -> None:
    """Смещение 0.01 SOH — это 1 процентный пункт, а не 0.01."""
    truth = _truth()
    result = evaluate(_submission_for(truth, error=0.01), truth, config=repo_config)
    assert result["overall"]["mae_pp"] == pytest.approx(1.0, abs=1e-6)
    assert result["overall"]["p95_abs_pp"] == pytest.approx(1.0, abs=1e-6)
    assert result["overall"]["bias_pp"] == pytest.approx(1.0, abs=1e-6)


def test_main_mode_ignores_lfp(repo_config: BenchmarkConfig) -> None:
    """§9: основной тест — только NCA и NMC, LFP в него не подмешивается."""
    truth = _truth()
    lfp = truth.filter(pl.col("chemistry") == "LFP")
    assert lfp.height > 0
    result = evaluate(_submission_for(truth), truth, config=repo_config)
    chemistries = {row["chemistry"] for row in result["by_chemistry"]}
    assert chemistries == {"NCA", "NMC"}
    assert result["n_truth"] == truth.height - lfp.height


def test_hard_mode_scores_only_lfp(repo_config: BenchmarkConfig) -> None:
    """§9: бонусная номинация LFP считается тем же eval.py с --hard-mode."""
    truth = _truth()
    lfp = truth.filter(pl.col("chemistry") == "LFP")
    result = evaluate(_submission_for(truth), truth, config=repo_config, hard_mode=True)
    assert result["mode"] == "hard"
    assert result["n_truth"] == lfp.height
    assert {row["chemistry"] for row in result["by_chemistry"]} == {"LFP"}


def test_chemistry_specific_error_is_visible(repo_config: BenchmarkConfig) -> None:
    """Разбивка по химии показывает, что модель сломалась на одной из них."""
    truth = _truth()
    submission = _submission_for(truth)
    submission = submission.with_columns(
        pl.when(
            pl.col("fragment_id").is_in(
                truth.filter(pl.col("chemistry") == "NMC")["fragment_id"].to_list()
            )
        )
        .then(pl.col("soh_pred") + 0.05)
        .otherwise(pl.col("soh_pred"))
        .alias("soh_pred")
    )
    result = evaluate(submission, truth, config=repo_config)
    by_chemistry = {row["chemistry"]: row for row in result["by_chemistry"]}
    assert by_chemistry["NCA"]["mae_pp"] == pytest.approx(0.0)
    assert by_chemistry["NMC"]["mae_pp"] == pytest.approx(5.0, abs=1e-6)
    assert result["overall"]["mae_pp"] > 0


def test_temperature_breakdown(repo_config: BenchmarkConfig) -> None:
    """Разбивка по температуре использует уставку камеры из cells (§4.1)."""
    truth = _truth()
    result = evaluate(_submission_for(truth), truth, config=repo_config)
    temperatures = {row["temperature_setpoint_c"] for row in result["by_temperature_setpoint_c"]}
    assert temperatures == {15.0, 25.0, 35.0}


def test_soh_deciles_cover_full_range(repo_config: BenchmarkConfig) -> None:
    """Разбивка по децилям SOH: сильно состаренные ячейки видны отдельно."""
    truth = _truth(n_chemistry=30)
    result = evaluate(_submission_for(truth, error=0.01), truth, config=repo_config)
    deciles = result["by_soh_decile"]
    assert len(deciles) == repo_config.eval.soh_deciles
    assert sum(row["n"] for row in deciles) == result["n_matched"]
    for row in deciles:
        assert "–" in row["soh_decile"]


def test_calibration_coverage_and_width(repo_config: BenchmarkConfig) -> None:
    """Калибровка: покрытие и средняя ширина интервала в п.п."""
    truth = _truth()
    covering = _submission_for(truth, width=0.05)
    result = evaluate(covering, truth, config=repo_config)
    assert result["calibration"]["coverage"] == pytest.approx(1.0)
    assert result["calibration"]["mean_width_pp"] == pytest.approx(10.0, abs=1e-6)

    # ошибка 0.03 при полуширине 0.01 — интервалы почти не покрывают истину
    narrow = _submission_for(truth, error=0.03, width=0.01)
    result = evaluate(narrow, truth, config=repo_config)
    assert result["calibration"]["coverage"] == pytest.approx(0.0)


def test_without_intervals_calibration_is_none(repo_config: BenchmarkConfig) -> None:
    """§11: без границ команда не участвует в номинации «калибровка»."""
    truth = _truth()
    result = evaluate(_submission_for(truth), truth, config=repo_config)
    assert result["calibration"] is None


def test_missing_and_extra_are_counted(repo_config: BenchmarkConfig) -> None:
    """Неполный сабмит и лишние строки видны в отчёте."""
    truth = _truth()
    submission = _submission_for(truth).filter(pl.col("fragment_id") != "f0")
    extra = pl.DataFrame({"fragment_id": ["zzz"], "soh_pred": [0.5]})
    submission = pl.concat([submission, extra], how="vertical")
    result = evaluate(submission, truth, config=repo_config)
    # f0 — единственный пропущенный оцениваемый фрагмент
    assert result["n_missing"] == 1
    # LFP-фрагменты в основном режиме не «лишние»: лишний только zzz (§9)
    assert result["n_extra"] == 1
    assert result["n_matched"] == result["n_truth"] - 1
    assert result["n_test_file"] == truth.height


def test_report_mentions_calibration_state(repo_config: BenchmarkConfig) -> None:
    """Таблица в stdout объясняет, доступна ли номинация «калибровка»."""
    truth = _truth()
    without = render_report(evaluate(_submission_for(truth), truth, config=repo_config))
    assert "калибровка: интервалы не заданы" in without
    with_intervals = render_report(
        evaluate(_submission_for(truth, width=0.05), truth, config=repo_config)
    )
    assert "покрытие 1.000" in with_intervals
    assert "MAE:" in with_intervals


def test_save_results_writes_json(tmp_path: Path, repo_config: BenchmarkConfig) -> None:
    """§10: результат пишется в results.json."""
    truth = _truth()
    result = evaluate(_submission_for(truth), truth, config=repo_config)
    path = save_results(result, tmp_path / "results.json")
    assert json.loads(path.read_text(encoding="utf-8"))["overall"]["mae_pp"] == pytest.approx(0.0)


def test_load_truth_joins_cell_metadata(
    make_config: Callable[[dict[str, Any] | None], BenchmarkConfig],
) -> None:
    """Эталон собирается из размеченных фрагментов и метаданных ячеек."""
    config = make_config()
    from bsb.schema import CELLS_SCHEMA, FRAGMENTS_SCHEMA, conform

    cells = pl.DataFrame(
        [
            {
                "cell_id": "sandia_NCA_1",
                "dataset": "sandia",
                "chemistry": "NCA",
                "form_factor": "cylindrical_18650",
                "nominal_capacity_ah": 3.2,
                "bol_capacity_ah": 3.1,
                "temperature_setpoint_c": 25.0,
                "charge_protocol": "CC-CV 0.5C",
                "batch_id": None,
                "notes": None,
            }
        ]
    )
    config.paths.unified_dir.mkdir(parents=True, exist_ok=True)
    conform(cells, CELLS_SCHEMA, name="cells").write_parquet(config.paths.unified_cells)

    fragments = pl.DataFrame(
        [
            {
                "fragment_id": "x1",
                "cell_id": "sandia_NCA_1",
                "cycle_index": 1,
                "v_start": 3.9,
                "v_end": 4.1,
                "q_norm": [0.1, 0.2],
                "current": [1.6, 1.6],
                "temperature": [25.0, 25.0],
                "soh": 0.97,
            }
        ]
    )
    test_path = config.paths.fragments_path("test", labeled=True)
    test_path.parent.mkdir(parents=True, exist_ok=True)
    conform(fragments, FRAGMENTS_SCHEMA, name="fragments").write_parquet(test_path)

    truth = load_truth(config)
    assert truth.height == 1
    assert truth["chemistry"].to_list() == ["NCA"]
    assert truth["temperature_setpoint_c"].to_list() == [25.0]


def test_load_truth_without_test_file_is_reported(
    make_config: Callable[[dict[str, Any] | None], BenchmarkConfig],
) -> None:
    """Без сгенерированного теста eval говорит, что делать."""
    with pytest.raises(FileNotFoundError, match="bsb fragments"):
        load_truth(make_config())


def test_run_eval_writes_results_to_config_path(
    make_config: Callable[[dict[str, Any] | None], BenchmarkConfig],
) -> None:
    """run_eval кладёт results.json по пути из конфига."""
    config = make_config()
    truth = _truth()
    from bsb.schema import FRAGMENTS_SCHEMA, conform

    test_path = config.paths.fragments_path("test", labeled=True)
    test_path.parent.mkdir(parents=True, exist_ok=True)
    frame = truth.select("fragment_id", "cell_id", "cycle_index", "soh").with_columns(
        pl.lit(3.9).alias("v_start"),
        pl.lit(4.1).alias("v_end"),
        pl.lit([0.1, 0.2]).alias("q_norm"),
        pl.lit([1.6, 1.6]).alias("current"),
        pl.lit([25.0, 25.0]).alias("temperature"),
    )
    conform(frame, FRAGMENTS_SCHEMA, name="fragments").write_parquet(test_path)
    # метаданные ячеек нужны для join
    from bsb.schema import CELLS_SCHEMA

    cells = pl.DataFrame(
        [
            {
                "cell_id": cell_id,
                "dataset": "sandia",
                "chemistry": chemistry,
                "form_factor": "cylindrical_18650",
                "nominal_capacity_ah": 3.2,
                "bol_capacity_ah": 3.1,
                "temperature_setpoint_c": 25.0,
                "charge_protocol": "CC-CV 0.5C",
                "batch_id": None,
                "notes": None,
            }
            for cell_id, chemistry in (
                (cell_id, chemistry)
                for cell_id, chemistry in zip(
                    truth["cell_id"].unique().to_list(),
                    truth.group_by("cell_id", maintain_order=True).first()["chemistry"].to_list(),
                    strict=True,
                )
            )
        ]
    )
    config.paths.unified_dir.mkdir(parents=True, exist_ok=True)
    conform(cells, CELLS_SCHEMA, name="cells").write_parquet(config.paths.unified_cells)

    result = run_eval(config, _submission_for(truth, error=0.02), results_path=None)
    written = json.loads(config.paths.results_path.read_text(encoding="utf-8"))
    assert written["overall"]["mae_pp"] == pytest.approx(2.0, abs=1e-6)
    assert result["nominal_coverage"] == pytest.approx(0.9)
    assert np.isfinite(result["overall"]["p95_abs_pp"])
