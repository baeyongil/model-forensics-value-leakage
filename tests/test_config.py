from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from model_forensics.config import load_preregistration, load_run_config

ROOT = Path(__file__).resolve().parents[1]


def test_primary_config_loads_and_requires_frozen_revisions() -> None:
    config = load_run_config(ROOT / "config/run_122b.yaml")
    assert config.model.id == "Qwen/Qwen3.5-122B-A10B"
    assert config.model.tensor_parallel_size == 8
    config.assert_execution_ready()
    unfrozen = config.model_copy(deep=True)
    unfrozen.model.revision = None
    with pytest.raises(ValueError, match="model revision"):
        unfrozen.assert_execution_ready()


def test_preregistration_has_exact_five_hour_total() -> None:
    config = load_run_config(ROOT / "config/run_122b.yaml")
    prereg = load_preregistration(config)
    assert prereg["investigation_time_minutes"]["total"] == 300
    components = {
        key: value for key, value in prereg["investigation_time_minutes"].items() if key != "total"
    }
    assert sum(components.values()) == 300


def test_preregistered_judge_routes_use_frontier_models_with_frozen_prices() -> None:
    config = load_run_config(ROOT / "config/run_122b.yaml")
    judging = load_preregistration(config)["external_judging"]
    assert judging["high_volume_outcome_and_trajectory"] == {
        "model": "anthropic/claude-opus-5",
        "input_usd_per_million_tokens": 5.0,
        "output_usd_per_million_tokens": 25.0,
        "rationale": "strongest_anthropic_route_for_all_primary_numeric_measurement",
    }
    assert [route["model"] for route in judging["semantic_classification_routes"]] == [
        "anthropic/claude-opus-5",
        "google/gemini-3.1-pro-preview",
    ]
    assert judging["outcome_calibration"]["scope"] == (
        "all_behavioral_and_resampling_final_outcomes"
    )


def test_budget_configuration_rejects_uncovered_categories(tmp_path) -> None:
    source = yaml.safe_load((ROOT / "config/smoke.yaml").read_text(encoding="utf-8"))
    source["execution"]["gpu_cost_hard_stop_usd"] = 10
    source["execution"]["api_cost_hard_stop_usd"] = 10
    source["execution"]["total_cost_hard_stop_usd"] = 15
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(source), encoding="utf-8")
    with pytest.raises(ValueError, match="total hard stop"):
        load_run_config(path)
