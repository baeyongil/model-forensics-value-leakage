from __future__ import annotations

import pandas as pd
import pytest

from model_forensics.analysis import (
    accuracy_anchor_lens_resampling_association,
    accuracy_neutral_movement_assessment,
    adjudicate_hypotheses,
    anchoring_assessments,
    apply_divergent_coverage_gate,
    behavior_missingness_summary,
    behavior_process_summary,
    behavior_stage_summary,
    behavior_timing_summary,
    behavioral_row_estimands,
    conservative_hypothesis_verdict,
    hypothesis_criterion_assessments,
    independent_task_direction_assessment,
    lens_signal_assessment,
    pooled_process_assessments,
    process_asymmetry_assessments,
    select_intervention_eligible_pairs,
    sentence_effect_table,
    validate_parse_rate,
    wilson_interval,
)


def test_wilson_interval_is_bounded() -> None:
    low, high = wilson_interval(5, 10)
    assert 0 <= low < 0.5 < high <= 1


def test_behavior_summary_uses_above_threshold_for_all_conditions() -> None:
    rows = [
        {
            "task": "giraffe",
            "condition": "baseline",
            "threshold": 100,
            "first_estimate": 90,
            "final_estimate": 110,
        },
        {
            "task": "giraffe",
            "condition": "baseline",
            "threshold": 100,
            "first_estimate": 120,
            "final_estimate": 130,
        },
    ]
    summary = behavior_stage_summary(rows)
    rates = dict(zip(summary["stage"], summary["rate"], strict=True))
    assert rates == {"first": 0.5, "final": 1.0}


def test_parse_rate_gate() -> None:
    rows = [{"final_estimate": 1} for _ in range(19)] + [{"final_estimate": None}]
    assert validate_parse_rate(rows) == 0.95


def test_direction_aligned_log_ratio_and_first_to_final_timing() -> None:
    rows = [
        {
            "task": "giraffe",
            "condition": "above_good",
            "threshold": 100,
            "first_estimate": 90,
            "final_estimate": 120,
        },
        {
            "task": "giraffe",
            "condition": "above_good",
            "threshold": 100,
            "first_estimate": 80,
            "final_estimate": 110,
        },
        {
            "task": "giraffe",
            "condition": "below_good",
            "threshold": 100,
            "first_estimate": 110,
            "final_estimate": 80,
        },
        {
            "task": "giraffe",
            "condition": "below_good",
            "threshold": 100,
            "first_estimate": 120,
            "final_estimate": 90,
        },
    ]
    estimands = behavioral_row_estimands(rows)
    assert (estimands["signed_log_ratio_first"] < 0).all()
    assert (estimands["signed_log_ratio_final"] > 0).all()
    assert estimands["good_side_change"].tolist() == [1, 1, 1, 1]

    timing = behavior_timing_summary(
        rows,
        bootstrap_replicates=100,
        permutation_replicates=100,
    )
    assert (timing["good_side_change"] == 1).all()
    assert (timing["signed_log_ratio_change"] > 0).all()
    assert set(timing["good_side_permutation_method"]) == {"exact_sign_flip"}


def test_missingness_summary_keeps_missing_as_its_own_outcome() -> None:
    rows = [
        {
            "task": "giraffe",
            "condition": "above_good",
            "threshold": 100,
            "first_estimate": 90,
            "final_estimate": None,
        },
        {
            "task": "giraffe",
            "condition": "above_good",
            "threshold": 100,
            "first_estimate": None,
            "final_estimate": 110,
        },
    ]
    summary = behavior_missingness_summary(rows)
    assert set(summary["outcome"]) == {"estimate_missing"}
    assert dict(zip(summary["stage"], summary["rate"], strict=True)) == {
        "first": 0.5,
        "final": 0.5,
    }


def test_process_summary_reports_revisions_crossing_and_stopping() -> None:
    rows = [
        {
            "task": "giraffe",
            "condition": "above_good",
            "first_estimate": 90,
            "final_estimate": 110,
            "revision_count": 2,
            "first_good_side": False,
            "final_good_side": True,
            "first_good_side_crossing_index": 1,
            "stopped_after_first_good_side_crossing": True,
            "revisions_after_good": 0,
        },
        {
            "task": "giraffe",
            "condition": "above_good",
            "first_estimate": 120,
            "final_estimate": 90,
            "revision_count": 3,
            "first_good_side": True,
            "final_good_side": False,
            "first_good_side_crossing_index": 0,
            "stopped_after_first_good_side_crossing": False,
            "revisions_after_good": 3,
        },
    ]
    row = behavior_process_summary(rows).iloc[0]
    assert row["revision_count_mean"] == 2.5
    assert row["reached_good_side_rate"] == 1
    assert row["stopped_after_first_good_side_crossing_rate"] == 0.5
    assert row["bad_to_good_rate"] == 0.5
    assert row["good_to_bad_rate"] == 0.5


def _sentence_rows() -> list[dict]:
    rows: list[dict] = []
    for class_name in ("value_threshold_planning", "accuracy_commitment"):
        for direction in ("above_good", "below_good"):
            for cluster_index in range(2):
                cluster = f"{class_name}-{direction}-{cluster_index}"
                for arm, good in (("retain", 1), ("resample", 0)):
                    rows.append(
                        {
                            "anchor_id": cluster,
                            "sentence_class": class_name,
                            "condition": direction,
                            "base_trace_id": cluster,
                            "arm": arm,
                            "sample_index": 0,
                            "seed": len(rows) // 2,
                            "stage": "initial",
                            "intervention_eligible": True,
                            "divergent": arm == "resample",
                            "final_good_side": good,
                            "threshold": 100,
                            "final_estimate": (
                                120 if (direction == "above_good") == bool(good) else 80
                            ),
                        }
                    )
    return rows


def test_sentence_table_has_one_confirmatory_contrast_and_exploratory_metadata() -> None:
    effects = sentence_effect_table(
        _sentence_rows(),
        bootstrap_replicates=100,
        permutation_replicates=100,
    )
    primary = effects[effects["is_confirmatory"]]
    assert primary["contrast_id"].tolist() == ["accuracy_commitment:pooled"]
    assert primary.iloc[0]["multiplicity_method"] == "none_single_confirmatory"
    exploratory = effects[~effects["is_confirmatory"]]
    assert set(exploratory["multiplicity_method"]) == {"holm"}
    assert exploratory["p_value_adjusted"].notna().all()
    assert (effects["worst_case_bound"] <= effects["best_case_bound"]).all()
    assert effects["signed_log_ratio_estimate"].notna().all()


@pytest.mark.parametrize(
    ("retain_value", "resample_value", "complete_conclusion", "expected_bound"),
    [
        (1, 0, "positive", (0.10, 1.0)),
        (0, 1, "negative", (-1.0, -0.10)),
        (0, 0, "practically_null", (-0.90, 0.0)),
    ],
)
def test_sentence_conclusion_is_blocked_when_missing_bounds_touch_or_cross_rope(
    retain_value: int,
    resample_value: int,
    complete_conclusion: str,
    expected_bound: tuple[float, float],
) -> None:
    rows = []
    for cluster_index in range(20):
        cluster = f"missing-bound-{cluster_index:02d}"
        rows.extend(
            [
                {
                    "sentence_class": "value_threshold_planning",
                    "condition": "above_good",
                    "base_trace_id": cluster,
                    "arm": "retain",
                    "final_good_side": retain_value,
                },
                {
                    "sentence_class": "value_threshold_planning",
                    "condition": "above_good",
                    "base_trace_id": cluster,
                    "arm": "resample",
                    "final_good_side": (
                        resample_value if cluster_index < 2 else None
                    ),
                },
            ]
        )

    effects = sentence_effect_table(
        rows,
        bootstrap_replicates=100,
        permutation_replicates=100,
        confirmatory_contrast=None,
    )
    pooled = effects[effects["direction"] == "pooled"].iloc[0]

    assert pooled["conclusion_before_missing_bounds_gate"] == complete_conclusion
    assert pooled["conclusion"] == "inconclusive"
    assert pooled["worst_case_bound"] == pytest.approx(expected_bound[0])
    assert pooled["best_case_bound"] == pytest.approx(expected_bound[1])
    assert not bool(pooled["missing_bounds_gate_passed"])
    assert pooled["missing_bounds_gate_state"] == "blocked_not_robust"
    assert pooled["missing_bounds_gate_audit"]["gated_conclusion"] == "inconclusive"


def test_divergent_coverage_gate_preserves_estimate_but_forces_inconclusive() -> None:
    rows = _sentence_rows()
    effects = sentence_effect_table(
        rows,
        bootstrap_replicates=100,
        permutation_replicates=100,
    )
    gated = apply_divergent_coverage_gate(effects, rows, minimum_per_anchor=2)

    assert gated["estimate"].notna().all()
    assert set(gated["conclusion_before_divergent_coverage_gate"]) == {"positive"}
    assert set(gated["conclusion"]) == {"inconclusive"}
    assert not gated["divergent_coverage_gate_passed"].any()
    assert set(gated["minimum_observed_eligible_divergent_resamples"]) == {1}


def test_intervention_pair_selection_preserves_resample_side_missingness() -> None:
    rows = _sentence_rows()
    target = next(
        row
        for row in rows
        if row["sentence_class"] == "accuracy_commitment"
        and row["condition"] == "above_good"
        and row["arm"] == "resample"
    )
    target["final_good_side"] = None
    target["final_estimate"] = None
    target["confirmatory_eligible"] = False

    selected = select_intervention_eligible_pairs(rows)
    selected_target = next(
        row for row in selected if row["seed"] == target["seed"] and row["arm"] == "resample"
    )
    assert selected_target["intervention_eligible"] is True
    assert selected_target["final_good_side"] is None

    effects = sentence_effect_table(
        selected,
        bootstrap_replicates=100,
        permutation_replicates=100,
    )
    pooled = effects[
        (effects["sentence_class"] == "accuracy_commitment") & (effects["direction"] == "pooled")
    ].iloc[0]
    assert pooled["resample_missing"] == 1
    assert pooled["resample_missing_rate"] > 0
    assert pooled["total_clusters"] == 4


def test_unfaithful_verdict_requires_causal_and_temporal_criteria_not_generic_lens() -> None:
    effects = pd.DataFrame(
        [
            {
                "sentence_class": "value_threshold_planning",
                "direction": "pooled",
                "conclusion": "positive",
                "inference_tier": "exploratory",
                "is_confirmatory": False,
            },
            {
                "sentence_class": "accuracy_commitment",
                "direction": "pooled",
                "conclusion": "practically_null",
                "inference_tier": "confirmatory",
                "is_confirmatory": True,
                "multiplicity_family": "single_predeclared_primary",
                "multiplicity_method": "none_single_confirmatory",
                "multiplicity_family_size": 1,
                "p_value": 0.25,
                "p_value_adjusted": 0.25,
            },
        ]
    )
    generic_lens_only = adjudicate_hypotheses(
        effects=effects,
        local_direction_gap_first=0,
        local_direction_gap_final=0.2,
        neutral_shift=0.2,
        coffee_same_sign=True,
        lens_corroborates=True,
    )
    post_hoc = next(
        verdict
        for verdict in generic_lens_only
        if verdict.hypothesis == "post_hoc_unfaithful_narration"
    )
    assert post_hoc.status == "not_established"
    assert "direction_signal_precedes_accuracy_statement" in post_hoc.unknown_criteria

    all_criteria = adjudicate_hypotheses(
        effects=effects,
        local_direction_gap_first=0,
        local_direction_gap_final=0.2,
        neutral_shift=0.2,
        coffee_same_sign=True,
        lens_corroborates=True,
        pre_statement_direction_signal=True,
        criterion_inference_tiers={"direction_signal_precedes_accuracy_statement": "observational"},
    )
    post_hoc = next(
        verdict for verdict in all_criteria if verdict.hypothesis == "post_hoc_unfaithful_narration"
    )
    assert post_hoc.status == "not_established"
    assert post_hoc.descriptive_criterion_values["value_planning_sentence_is_load_bearing"] is True
    assert post_hoc.criterion_values["value_planning_sentence_is_load_bearing"] is None
    assert "value_planning_sentence_is_load_bearing" in post_hoc.nonconfirmatory_criteria


def test_nonconfirmatory_criterion_cannot_formally_support_a_hypothesis() -> None:
    verdict = conservative_hypothesis_verdict(
        "audit_guard",
        criteria={"primary": True, "exploratory_screen": True},
        evidence="both descriptive estimates are positive",
        boundary="Only the primary is confirmatory.",
        criterion_inference_tiers={
            "primary": "confirmatory",
            "exploratory_screen": "exploratory",
        },
    )

    assert verdict.status == "not_established"
    assert verdict.criterion_values == {"primary": True, "exploratory_screen": None}
    assert verdict.descriptive_criterion_values == {
        "primary": True,
        "exploratory_screen": True,
    }
    assert verdict.nonconfirmatory_criteria == ("exploratory_screen",)


def test_confirmatory_effect_requires_consistent_multiplicity_metadata() -> None:
    effects = pd.DataFrame(
        [
            {
                "sentence_class": "accuracy_commitment",
                "direction": "pooled",
                "conclusion": "negative",
                "inference_tier": "confirmatory",
                "is_confirmatory": True,
                "multiplicity_family": "single_predeclared_primary",
                "multiplicity_method": "holm",
                "multiplicity_family_size": 1,
                "p_value": 0.01,
                "p_value_adjusted": 0.01,
            }
        ]
    )

    with pytest.raises(ValueError, match="inconsistent multiplicity metadata"):
        adjudicate_hypotheses(
            effects=effects,
            local_direction_gap_first=None,
            local_direction_gap_final=None,
            neutral_shift=None,
            coffee_same_sign=None,
            lens_corroborates=None,
        )


def _lens_rows(*, traces: tuple[str, ...], negative: bool = False) -> list[dict]:
    rows: list[dict] = []
    sign = -1.0 if negative else 1.0
    for trace_index, trace_id in enumerate(traces):
        for lens_type, scale in (("j", 1.0), ("r", 0.8)):
            for layer in (4, 19, 33):
                for position, position_scale in (
                    ("first_estimate_pre", 1.0),
                    ("anchor_pre", 1.2),
                    ("anchor_post", 1.7),
                    ("final_answer_pre", 2.0),
                ):
                    for concept_set in ("direction", "epistemic"):
                        value = sign * scale * position_scale + trace_index * 0.001
                        if concept_set == "epistemic":
                            value *= 0.5
                        rows.append(
                            {
                                "trace_id": trace_id,
                                "lens_type": lens_type,
                                "layer": layer,
                                "position": position,
                                "concept_set": concept_set,
                                "signed_contrast": value,
                                "causal_claim": False,
                            }
                        )
    return rows


def test_lens_signal_requires_jr_uncertainty_and_adjacent_layer_bands() -> None:
    positive = lens_signal_assessment(
        _lens_rows(traces=("a", "b", "c", "d")),
        criterion="direction_signal_present_before_first_estimate",
        concept_set="direction",
        position="first_estimate_pre",
        bootstrap_replicates=100,
    )
    assert positive.value is True
    assert positive.details["adjacent_positive_pairs"] == [
        ["early", "middle"],
        ["middle", "late"],
    ]
    assert positive.details["lens_is_observational_only"] is True

    contrary = lens_signal_assessment(
        _lens_rows(traces=("a", "b", "c", "d"), negative=True),
        criterion="direction_signal_present_before_first_estimate",
        concept_set="direction",
        position="first_estimate_pre",
        bootstrap_replicates=100,
    )
    assert contrary.value is False
    assert contrary.details["adjacent_negative_pairs"]


def _accuracy_association_rows(*, missing: bool = False) -> tuple[list[dict], list[dict]]:
    resampling: list[dict] = []
    lens: list[dict] = []
    layers = range(4, 47)
    for direction in ("above_good", "below_good"):
        for trace_index, resample_good_count in enumerate((0, 2, 4, 6)):
            trace_id = f"{direction}-{trace_index}"
            for sample_index in range(8):
                for arm in ("retain", "resample"):
                    outcome: int | None = int(
                        arm == "resample" and sample_index < resample_good_count
                    )
                    if missing and trace_index == 0 and sample_index == 0 and arm == "resample":
                        outcome = None
                    resampling.append(
                        {
                            "anchor_id": trace_id,
                            "base_trace_id": trace_id,
                            "sentence_class": "accuracy_commitment",
                            "condition": direction,
                            "sample_index": sample_index,
                            "seed": sample_index,
                            "stage": "initial",
                            "arm": arm,
                            "intervention_eligible": True,
                            "divergent": arm == "resample",
                            "final_good_side": outcome,
                        }
                    )
            change = resample_good_count / 8
            for lens_type, scale in (("J", 1.0), ("R", 0.8)):
                for layer in layers:
                    for position, value in (("anchor_pre", 0.0), ("anchor_post", change)):
                        lens.append(
                            {
                                "trace_id": trace_id,
                                "lens_type": lens_type,
                                "layer": layer,
                                "position_name": position,
                                "contrast": "epistemic",
                                "signed_mean_logit_contrast": value * scale,
                                "probe_eligible": True,
                            }
                        )
    return resampling, lens


def test_accuracy_lens_association_uses_exact_stratified_permutations_and_equal_bands() -> None:
    resampling, lens = _accuracy_association_rows()
    result = accuracy_anchor_lens_resampling_association(resampling, lens)
    assert result["status"] == "available"
    assert result["common_trace_count"] == 8
    assert result["permutation_count"] == 576
    assert result["permutation_resolution"] == pytest.approx(1 / 576)
    assert result["per_lens"]["J"]["tau_a"] == 1
    assert result["per_lens"]["R"]["tau_a"] == 1
    assert result["causal_claim"] is False
    assert result["lens_weights"] == {"early": 1 / 3, "middle": 1 / 3, "late": 1 / 3}


def test_accuracy_lens_association_fails_closed_on_an_eligible_missing_outcome() -> None:
    resampling, lens = _accuracy_association_rows(missing=True)
    result = accuracy_anchor_lens_resampling_association(resampling, lens)
    assert result["status"] == "unavailable"
    assert result["reason"] == "eligible paired resampling outcomes are missing"
    affected = next(row for row in result["trace_effects"] if row["trace_id"].endswith("-0"))
    assert affected["d_i"] is None
    assert affected["d_i_lower"] < affected["d_i_upper"]


def test_accuracy_movement_uses_direction_matched_threshold_only_comparator() -> None:
    resampling: list[dict] = []
    for direction, retain, resample in (
        ("above_good", 0, 1),
        ("below_good", 1, 0),
    ):
        for trace_index in range(3):
            for arm, outcome in (("retain", retain), ("resample", resample)):
                resampling.append(
                    {
                        "sentence_class": "accuracy_commitment",
                        "base_trace_id": f"{direction}-{trace_index}",
                        "condition": direction,
                        "arm": arm,
                        "final_good_side": outcome,
                    }
                )
    controls = [
        {
            "task": "giraffe",
            "condition": "threshold_only",
            "threshold": 100,
            "final_estimate": 90,
        }
        for _ in range(6)
    ]
    assessment = accuracy_neutral_movement_assessment(
        resampling,
        controls,
        bootstrap_replicates=100,
    )
    assert assessment.value is True
    assert assessment.estimate == -1
    assert assessment.details["neutral_above_rate"] == 0
    assert assessment.details["neutral_below_rate"] == 1


def test_anchoring_and_process_estimands_have_fixed_three_state_rules() -> None:
    rollouts: list[dict] = []
    condition_scales = {
        "baseline": (90, 110),
        "threshold_only": (110, 110),
        "above_good": (110, 110),
        "below_good": (90, 90),
    }
    for condition, values in condition_scales.items():
        for index in range(8):
            final = values[index % len(values)]
            rollouts.append(
                {
                    "task": "giraffe",
                    "condition": condition,
                    "threshold": 100,
                    "final_estimate": final,
                    "revision_count": 3 if condition == "above_good" else 1,
                    "first_good_side_crossing_index": 0,
                    "stopped_after_first_good_side_crossing": (condition == "above_good"),
                }
            )
    anchoring = {
        item.criterion: item for item in anchoring_assessments(rollouts, bootstrap_replicates=100)
    }
    assert anchoring["threshold_only_shift_is_material"].value is True
    assert anchoring["moral_direction_interaction_is_practically_weak"].value is False

    process = {
        item.criterion: item
        for item in process_asymmetry_assessments(rollouts, bootstrap_replicates=100)
    }
    assert process["revision_pattern_is_direction_asymmetric"].value is True
    assert process["good_side_stopping_is_direction_asymmetric"].value is True


def test_stopping_primary_estimand_pools_condition_relative_good_side() -> None:
    rollouts = []
    for condition in ("above_good", "below_good"):
        for _ in range(20):
            rollouts.append(
                {
                    "task": "giraffe",
                    "condition": condition,
                    "first_good_side": 0,
                    "final_good_side": 1,
                    "revision_count": 1,
                    "first_good_side_crossing_index": 1,
                    "stopped_after_first_good_side_crossing": True,
                    "trajectory_measurement_valid": True,
                }
            )

    pooled = {
        item.criterion: item
        for item in pooled_process_assessments(rollouts, bootstrap_replicates=100)
    }
    assert pooled["pooled_good_side_revision_is_positive"].value is True
    assert pooled["pooled_stopping_after_good_crossing_is_prevalent"].value is True
    assert pooled["pooled_good_side_revision_is_positive"].estimate == 1.0
    assert pooled["pooled_stopping_after_good_crossing_is_prevalent"].estimate == 0.5
    assert (
        pooled["pooled_good_side_revision_is_positive"].details[
            "direction_heterogeneity_is_exploratory"
        ]
        is True
    )


def test_behavioral_missing_bounds_block_complete_case_direction_claim() -> None:
    rollouts = []
    # Complete cases differ by 2pp with enough precision for a positive CI,
    # while exactly 5% missing above-good outcomes permit a negative assignment.
    for index in range(10_000):
        if index < 500:
            estimate = None
        else:
            estimate = 110 if index - 500 < 4_940 else 90
        rollouts.append(
            {
                "task": "chicago_coffee",
                "condition": "above_good",
                "threshold": 100,
                "final_estimate": estimate,
            }
        )
    for index in range(10_000):
        rollouts.append(
            {
                "task": "chicago_coffee",
                "condition": "below_good",
                "threshold": 100,
                "final_estimate": 110 if index < 5_000 else 90,
            }
        )

    assessment = independent_task_direction_assessment(
        rollouts,
        bootstrap_replicates=100,
    )
    assert assessment.estimate == pytest.approx(0.02)
    assert assessment.ci_low is not None and assessment.ci_low > 0
    assert assessment.value is None
    assert "best/worst missing-data bounds" in assessment.reason
    assert assessment.details["task_condition_quality_gate_passed"] is True
    assert assessment.details["missing_assignment_bound_low"] < 0
    assert assessment.details["missing_assignment_bound_high"] > 0


def test_full_criterion_builder_and_verdicts_emit_unknown_reasons() -> None:
    traces = tuple(f"trace-{index}" for index in range(4))
    resampling = []
    for trace_id in traces:
        for arm, good in (("retain", 0), ("resample", 1)):
            resampling.append(
                {
                    "base_trace_id": trace_id,
                    "sentence_class": "accuracy_commitment",
                    "condition": "above_good",
                    "arm": arm,
                    "final_good_side": good,
                }
            )
    rollouts = [
        {
            "task": "giraffe",
            "condition": "threshold_only",
            "threshold": 100,
            "first_estimate": 90,
            "final_estimate": 90,
        }
        for _ in range(4)
    ]
    criteria = hypothesis_criterion_assessments(
        rollout_rows=rollouts,
        resampling_rows=resampling,
        primary_resampling_rows=resampling,
        lens_rows=_lens_rows(traces=traces),
        bootstrap_replicates=100,
    )
    by_name = {item.criterion: item for item in criteria}
    assert by_name["direction_signal_precedes_accuracy_statement"].value is True
    assert by_name["direction_signal_precedes_accuracy_statement"].inference_tier == (
        "observational"
    )
    assert by_name["objective_signal_increases_after_accuracy_sentence"].value is True
    assert by_name["accuracy_sentence_moves_toward_neutral_baseline"].inference_tier == (
        "supportive"
    )
    assert by_name["independent_task_same_direction"].value is None
    assert by_name["independent_task_same_direction"].inference_tier == "exploratory"
    assert "requires at least two" in by_name["independent_task_same_direction"].reason

    effects = pd.DataFrame(
        [
            {
                "sentence_class": "value_threshold_planning",
                "direction": "pooled",
                "conclusion": "inconclusive",
            },
            {
                "sentence_class": "accuracy_commitment",
                "direction": "pooled",
                "conclusion": "inconclusive",
            },
        ]
    )
    verdict = adjudicate_hypotheses(
        effects=effects,
        local_direction_gap_first=None,
        local_direction_gap_final=None,
        neutral_shift=None,
        coffee_same_sign=None,
        lens_corroborates=None,
    )[0]
    assert verdict.status == "not_established"
    assert verdict.criterion_values["value_planning_sentence_causal_effect_positive"] is None
    assert (
        verdict.unknown_criterion_reasons["value_planning_sentence_causal_effect_positive"]
        == "pooled value-planning sentence conclusion=inconclusive"
    )
