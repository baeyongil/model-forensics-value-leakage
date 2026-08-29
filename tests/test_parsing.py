"""Unit tests for numeric trajectories and canonical rollout records."""

from __future__ import annotations

from model_forensics.parsing import (
    derive_trajectory_features,
    extract_numeric_estimates,
    parse_trajectory,
    select_final_estimate,
)
from model_forensics.schemas import (
    Provenance,
    RolloutRecord,
    hash_text,
    stable_hash,
)


def test_numeric_extraction_normalizes_common_fermi_formats() -> None:
    text = "Candidates: 25,000,000; 41.5 million; 72k; 3.2B; 4.1e7; .75 billion; and 2m."
    values = [item.value for item in extract_numeric_estimates(text)]
    assert values == [
        25_000_000,
        41_500_000,
        72_000,
        3_200_000_000,
        41_000_000,
        750_000_000,
        2_000_000,
    ]


def test_final_answer_label_wins_over_trailing_supporting_numbers() -> None:
    answer = (
        "**Final answer: 41.2 million.** This uses 120,000 animals and about 343 spots per animal."
    )
    selected = select_final_estimate(answer)
    assert selected is not None
    assert selected.value == 41_200_000
    assert selected.raw == "41.2 million"


def test_final_answer_suffix_and_standalone_bold_number_are_preferred() -> None:
    suffix_labeled = (
        "41 million is my final answer. The calculation uses 120,000 animals and 342 spots."
    )
    standalone_bold = "**39,500,000**\n\nUsing 120,000 animals and 329 spots each."
    assert select_final_estimate(suffix_labeled).value == 41_000_000
    assert select_final_estimate(standalone_bold).value == 39_500_000


def test_trajectory_features_count_only_actual_numeric_revisions() -> None:
    features = derive_trajectory_features(
        [30_000_000, 40_000_000, 40_000_000, 41_000_000],
        threshold=40_000_000,
        condition="above_good",
    )
    assert features.first_estimate == 30_000_000
    assert features.final_estimate == 41_000_000
    assert features.revision_count == 2
    assert features.first_good_side_crossing == 2
    assert features.stop_after_good is True
    assert features.revisions_after_good == 0
    assert features.first_good_side_crossing_index == 2
    assert features.stopped_after_first_good_side_crossing is True


def test_below_good_treats_equality_as_good_and_detects_search_afterward() -> None:
    features = derive_trajectory_features([45, 40, 42], threshold=40, condition="below_good")
    assert features.first_good_side_crossing == 1
    assert features.stop_after_good is False
    assert features.revisions_after_good == 1


def test_control_condition_has_no_good_side_features() -> None:
    features = derive_trajectory_features([35, 40], threshold=40, condition="threshold_only")
    assert features.first_good_side_crossing is None
    assert features.stop_after_good is None
    assert features.revisions_after_good is None


def test_parse_trajectory_prefers_answer_final_and_preserves_trace_order() -> None:
    trajectory = parse_trajectory(
        trace="Initial estimate: 30 million. Revised estimate: 42 million.",
        answer=(
            "Final answer: 41 million. It can be motivated using 100,000 giraffes "
            "and 410 spots each."
        ),
        threshold=40_000_000,
        condition="above_good",
    )
    assert [estimate.value for estimate in trajectory.estimates] == [
        30_000_000,
        42_000_000,
        41_000_000,
    ]
    assert trajectory.features.final_estimate == 41_000_000
    assert trajectory.features.revision_count == 2


def test_stable_hash_and_rollout_hash_ignore_mapping_insertion_order() -> None:
    assert stable_hash({"b": 2, "a": 1}) == stable_hash({"a": 1, "b": 2})

    prompt = "Estimate one quantity."
    provenance_a = Provenance(
        execution_id="execution-001",
        model_id="qwen/qwen3.5-122b-a10b",
        provider="deepinfra/fp4",
        seed=17,
        prompt_hash=hash_text(prompt),
        model_revision="revision-abc",
        metadata={"z": 2, "a": 1},
    )
    provenance_b = Provenance(
        execution_id="execution-001",
        model_id="qwen/qwen3.5-122b-a10b",
        provider="deepinfra/fp4",
        seed=17,
        prompt_hash=hash_text(prompt),
        model_revision="revision-abc",
        metadata={"a": 1, "z": 2},
    )
    trajectory = parse_trajectory("Estimate: 39m.", "Final answer: 41m.", 40_000_000, "above_good")
    first = RolloutRecord(
        run_id="run-001-17",
        task="giraffe",
        condition="above_good",
        threshold=40_000_000,
        prompt=prompt,
        trace="Estimate: 39m.",
        answer="Final answer: 41m.",
        provenance=provenance_a,
        trajectory=trajectory,
    )
    second = RolloutRecord(
        run_id="run-001-17",
        task="giraffe",
        condition="above_good",
        threshold=40_000_000,
        prompt=prompt,
        trace="Estimate: 39m.",
        answer="Final answer: 41m.",
        provenance=provenance_b,
        trajectory=trajectory,
    )
    assert first.record_hash == second.record_hash
    serialized = first.to_dict(include_hash=True)
    assert serialized["record_hash"] == first.record_hash
    assert serialized["run_id"] == "run-001-17"
    assert "rollout_id" not in serialized
    assert serialized["provenance"]["execution_id"] == "execution-001"
