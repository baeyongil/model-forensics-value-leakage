from __future__ import annotations

import pytest

from model_forensics.statistics import (
    ci_half_width,
    classify_effect,
    cluster_missingness_sensitivity,
    cluster_risk_difference,
    holm_adjust_pvalues,
    paired_cluster_permutation,
    signed_log_ratio,
)


def _rows(cluster_effects: list[tuple[float, float]]) -> list[dict]:
    rows = []
    for index, (retain, resample) in enumerate(cluster_effects):
        rows.extend(
            [
                {"base_trace_id": f"t{index}", "arm": "retain", "final_good_side": retain},
                {"base_trace_id": f"t{index}", "arm": "resample", "final_good_side": resample},
            ]
        )
    return rows


def test_cluster_risk_difference_uses_base_trace_effects() -> None:
    estimate = cluster_risk_difference(
        _rows([(1, 0), (1, 0), (1, 0), (1, 0)]),
        bootstrap_replicates=250,
        permutation_replicates=250,
        seed=3,
    )
    assert estimate.estimate == 1
    assert estimate.clusters == 4
    assert estimate.conclusion == "positive"
    assert estimate.permutation_method == "exact_sign_flip"
    assert estimate.p_value == pytest.approx(0.125)


def test_incomplete_cluster_is_rejected() -> None:
    with pytest.raises(ValueError, match="lack both arms"):
        cluster_risk_difference(
            [
                {"base_trace_id": "a", "arm": "retain", "final_good_side": 1},
                {"base_trace_id": "b", "arm": "retain", "final_good_side": 1},
                {"base_trace_id": "b", "arm": "resample", "final_good_side": 0},
            ],
            bootstrap_replicates=20,
            permutation_replicates=20,
        )


def test_rope_and_signed_log_helpers() -> None:
    assert classify_effect(-0.05, 0.05) == "practically_null"
    assert classify_effect(0.11, 0.20) == "positive"
    assert classify_effect(-0.20, -0.11) == "negative"
    assert classify_effect(0.05, 0.15) == "inconclusive"
    assert ci_half_width(-0.2, 0.4) == pytest.approx(0.3)
    assert signed_log_ratio(200, 100, 1) > 0
    assert signed_log_ratio(200, 100, -1) < 0
    assert signed_log_ratio(50, 100, -1) > 0


def test_sign_flip_is_exact_when_enumeration_is_feasible() -> None:
    result = paired_cluster_permutation([1, 1, 1, 1], replicates=7, seed=999)
    assert result.method == "exact_sign_flip"
    assert result.assignments == 16
    assert result.p_value == pytest.approx(2 / 16)


def test_sign_flip_records_monte_carlo_method_when_exact_is_disabled() -> None:
    result = paired_cluster_permutation(
        [1, 0.5, -0.1, 0.2],
        replicates=101,
        seed=7,
        exact_max_assignments=2,
    )
    assert result.method == "monte_carlo_sign_flip"
    assert result.assignments == 101
    assert 0 < result.p_value <= 1


def test_missingness_is_an_outcome_and_bounds_are_logical_extremes() -> None:
    rows = [
        {"base_trace_id": "a", "arm": "retain", "final_good_side": 1},
        {"base_trace_id": "a", "arm": "retain", "final_good_side": None},
        {"base_trace_id": "a", "arm": "resample", "final_good_side": 0},
        {"base_trace_id": "a", "arm": "resample", "final_good_side": None},
        {"base_trace_id": "b", "arm": "retain", "final_good_side": 1},
        {"base_trace_id": "b", "arm": "retain", "final_good_side": 1},
        {"base_trace_id": "b", "arm": "resample", "final_good_side": 0},
        {"base_trace_id": "b", "arm": "resample", "final_good_side": 0},
    ]
    sensitivity = cluster_missingness_sensitivity(
        rows,
        bootstrap_replicates=100,
        permutation_replicates=100,
        seed=3,
    )
    assert sensitivity.complete_case is not None
    assert sensitivity.complete_case.estimate == 1
    assert sensitivity.retain_missing_rate == pytest.approx(0.25)
    assert sensitivity.resample_missing_rate == pytest.approx(0.25)
    assert sensitivity.missingness_effect is not None
    assert sensitivity.missingness_effect.estimate == 0
    assert sensitivity.worst_case_bound == pytest.approx(0.5)
    assert sensitivity.best_case_bound == pytest.approx(1.0)
    assert sensitivity.worst_case_bound <= sensitivity.complete_case.estimate
    assert sensitivity.complete_case.estimate <= sensitivity.best_case_bound


def test_complete_case_clusters_are_reported_when_one_cluster_has_an_empty_observed_arm() -> None:
    rows = [
        {"base_trace_id": "a", "arm": "retain", "final_good_side": None},
        {"base_trace_id": "a", "arm": "resample", "final_good_side": 0},
        {"base_trace_id": "b", "arm": "retain", "final_good_side": 1},
        {"base_trace_id": "b", "arm": "resample", "final_good_side": 0},
        {"base_trace_id": "c", "arm": "retain", "final_good_side": 1},
        {"base_trace_id": "c", "arm": "resample", "final_good_side": 0},
    ]
    sensitivity = cluster_missingness_sensitivity(
        rows,
        bootstrap_replicates=50,
        permutation_replicates=50,
    )
    assert sensitivity.total_clusters == 3
    assert sensitivity.complete_case_clusters == 2
    assert sensitivity.excluded_from_complete_case_clusters == 1
    assert sensitivity.complete_case is not None


def test_holm_adjustment_is_monotone_in_sorted_p_values() -> None:
    adjusted = holm_adjust_pvalues([0.04, None, 0.01, 0.03])
    assert adjusted == pytest.approx([0.06, None, 0.03, 0.06], nan_ok=True)
