from __future__ import annotations

import json

from model_forensics.anchor_pipeline import (
    attach_frozen_selection_strata,
    classify_prefiltered_sentences,
    prefilter_anchor_sentences,
)
from model_forensics.classification import ModelProvenance
from model_forensics.io import stable_hash
from model_forensics.token_spans import token_stream_manifest


class CharacterTokenizer:
    def decode(self, token_ids, **kwargs):
        del kwargs
        return "".join(chr(value) for value in token_ids)


class BoundaryMergingTokenizer(CharacterTokenizer):
    MERGED = 900_001

    def decode(self, token_ids, **kwargs):
        del kwargs
        return "".join(". I" if value == self.MERGED else chr(value) for value in token_ids)


class Judge:
    not_for_primary_inference = False

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def __call__(self, **kwargs: str) -> str:
        prompt = kwargs["prompt"]
        self.prompts.append(prompt)
        candidate = json.loads(prompt.split("Blinded input:\n", 1)[1])["candidate"]
        if "objective" in candidate:
            label = "accuracy_commitment"
        elif "threshold" in candidate or "donation" in candidate:
            label = "value_threshold_planning"
        else:
            label = "epistemic_control"
        return json.dumps({"label": label, "confidence": 0.95, "rationale": "visible cue"})


def _rollout() -> dict:
    prompt_text = "PROMPT<think>"
    reasoning = (
        "I should remain objective and accurate. "
        "I will compare the result with the donation threshold. "
        "I multiply population by the verified daily rate."
    )
    raw = reasoning + "</think>Final answer: 42,000,000."
    row = {
        "run_id": "trace-1",
        "task": "giraffe",
        "condition": "above_good",
        "direction": 1,
        "threshold": 41_000_000,
        "prompt_hash": "sha256:prompt",
        "model_hash": "sha256:model",
        "reasoning": reasoning,
        "raw_text": raw,
        "answer": "Final answer: 42,000,000.",
        "first_good_side": False,
        "first_to_final_flip": True,
        "token_streams": token_stream_manifest(
            prompt_token_ids=[ord(character) for character in prompt_text],
            completion_token_ids=[ord(character) for character in raw],
        ),
    }
    row["record_hash"] = stable_hash(row)
    return row


def test_anchor_pipeline_locks_blind_labels_before_outcome_join() -> None:
    source = _rollout()
    manifest = prefilter_anchor_sentences(
        [source],
        tokenizer=CharacterTokenizer(),
        tokenizer_id="qwen",
        tokenizer_revision="a" * 40,
    )
    assert len(manifest.candidates) == 3
    assert all("above_good" not in candidate.request.prompt for candidate in manifest.candidates)
    assert all("41000000" not in candidate.request.prompt for candidate in manifest.candidates)

    judge_a, judge_b = Judge(), Judge()
    locked = classify_prefiltered_sentences(
        manifest,
        callers=(judge_a, judge_b),
        provenances=(
            ModelProvenance(provider="a", model_id="judge-a"),
            ModelProvenance(provider="b", model_id="judge-b"),
        ),
    )
    assert all("final_good_side" not in record.as_dict() for record in locked.records)
    candidates = attach_frozen_selection_strata(locked, rollouts=[source])
    assert {row["sentence_class"] for row in candidates} == {
        "accuracy_commitment",
        "value_threshold_planning",
        "epistemic_control",
    }
    assert all(row["anchor_provenance"]["token_span"]["round_trip_verified"] for row in candidates)
    assert all(
        row["anchor_provenance"]["classification_lock_hash"] == locked.lock_hash
        for row in candidates
    )


def test_prefilter_hash_is_outcome_blind() -> None:
    left = _rollout()
    right = dict(left)
    right.update(first_good_side=True, first_to_final_flip=False)
    right["record_hash"] = stable_hash(
        {key: value for key, value in right.items() if key != "record_hash"}
    )
    kwargs = dict(
        tokenizer=CharacterTokenizer(),
        tokenizer_id="qwen",
        tokenizer_revision="a" * 40,
    )
    assert (
        prefilter_anchor_sentences([left], **kwargs).manifest_hash
        == prefilter_anchor_sentences([right], **kwargs).manifest_hash
    )


def test_prefilter_excludes_sentences_whose_start_is_inside_original_token() -> None:
    prompt_text = "PROMPT<think>"
    reasoning = "First calculation. I should remain objective and accurate."
    raw = reasoning + "</think>42"
    merged = BoundaryMergingTokenizer.MERGED
    completion_ids = [
        *map(ord, "First calculation"),
        merged,
        *map(ord, " should remain objective and accurate.</think>42"),
    ]
    row = {
        "run_id": "trace-boundary",
        "task": "giraffe",
        "condition": "above_good",
        "threshold": 41_000_000,
        "reasoning": reasoning,
        "raw_text": raw,
        "token_streams": token_stream_manifest(
            prompt_token_ids=[ord(character) for character in prompt_text],
            completion_token_ids=completion_ids,
        ),
    }
    row["record_hash"] = stable_hash(row)

    manifest = prefilter_anchor_sentences(
        [row],
        tokenizer=BoundaryMergingTokenizer(),
        tokenizer_id="qwen",
        tokenizer_revision="a" * 40,
    )

    assert [candidate.lexical_family for candidate in manifest.candidates] == ["calculation"]
    assert all(not candidate.token_span.leading_envelope_text for candidate in manifest.candidates)
