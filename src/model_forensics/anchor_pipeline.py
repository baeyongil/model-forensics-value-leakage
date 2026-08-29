"""Outcome-blind anchor classification followed by hash-gated strata attachment."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from model_forensics.anchors import DEFAULT_SENTENCE_CLASSES, SentenceSpan, sentence_spans
from model_forensics.classification import (
    ClassificationCaller,
    ClassificationRequest,
    ClassifiedCandidateRecord,
    ModelProvenance,
    build_blinded_request,
    classify_primary,
    lock_candidate_record,
)
from model_forensics.io import assert_unique, stable_hash
from model_forensics.token_spans import CompletionTokenMap, OriginalTokenSpan

PREFILTER_VERSION = "outcome-blind-anchor-prefilter-v1"

_FAMILY_PATTERNS = {
    "accuracy": re.compile(
        r"\b(?:accur|objective|unbias|bias|sway|regardless|independent)\w*\b",
        re.IGNORECASE,
    ),
    "value": re.compile(
        r"\b(?:threshold|donat|charity|benefi|harm|good cause|bad cause|above|below|"
        r"target|cross|stop)\w*\b",
        re.IGNORECASE,
    ),
    "calculation": re.compile(
        r"\b(?:calculat|multiply|divide|population|rate|fraction|share|total|per day|"
        r"estimate|compute|verify|check)\w*\b",
        re.IGNORECASE,
    ),
}


class AnchorPipelineError(RuntimeError):
    """A classification or join invariant would make anchor selection ambiguous."""


@dataclass(frozen=True)
class PrefilteredSentence:
    trace_id: str
    source_reasoning: str
    request: ClassificationRequest
    token_span: OriginalTokenSpan
    lexical_family: str

    def blind_dict(self) -> dict[str, Any]:
        """Serialize without condition, direction, threshold, estimates, or answer."""

        return {
            "trace_id": self.trace_id,
            "request": self.request.audit_dict(),
            "token_span": self.token_span.as_dict(),
            "lexical_family": self.lexical_family,
            "source_reasoning_hash": stable_hash(self.source_reasoning),
        }


@dataclass(frozen=True)
class AnchorPrefilterManifest:
    candidates: tuple[PrefilteredSentence, ...]
    tokenizer_id: str
    tokenizer_revision: str
    max_per_trace_per_family: int
    manifest_hash: str
    version: str = PREFILTER_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "tokenizer_id": self.tokenizer_id,
            "tokenizer_revision": self.tokenizer_revision,
            "max_per_trace_per_family": self.max_per_trace_per_family,
            "candidates": [candidate.blind_dict() for candidate in self.candidates],
            "manifest_hash": self.manifest_hash,
        }


@dataclass(frozen=True)
class LockedClassifications:
    records: tuple[ClassifiedCandidateRecord, ...]
    token_spans: Mapping[str, OriginalTokenSpan]
    prefilter_manifest_hash: str
    lock_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "prefilter_manifest_hash": self.prefilter_manifest_hash,
            "records": [record.as_dict() for record in self.records],
            "token_spans": {
                candidate_id: span.as_dict()
                for candidate_id, span in sorted(self.token_spans.items())
            },
            "lock_hash": self.lock_hash,
        }


def _validate_source_rollout(row: Mapping[str, Any]) -> None:
    record_hash = row.get("record_hash")
    if not isinstance(record_hash, str):
        raise AnchorPipelineError("rollout lacks a content hash")
    unhashed = {key: value for key, value in row.items() if key != "record_hash"}
    if stable_hash(unhashed) != record_hash:
        raise AnchorPipelineError(f"rollout hash mismatch: {row.get('run_id')}")


def _lexical_family(span: SentenceSpan) -> str | None:
    matches = [name for name, pattern in _FAMILY_PATTERNS.items() if pattern.search(span.text)]
    if not matches:
        return None
    # Accuracy/value cues take priority over arithmetic cues when one sentence
    # contains both. The external judges still assign the frozen final label.
    for family in ("accuracy", "value", "calculation"):
        if family in matches:
            return family
    raise AssertionError("unreachable lexical family")


def prefilter_anchor_sentences(
    rollouts: Sequence[Mapping[str, Any]],
    *,
    tokenizer: Any,
    tokenizer_id: str,
    tokenizer_revision: str,
    max_per_trace_per_family: int = 2,
) -> AnchorPrefilterManifest:
    """Freeze bounded, outcome-blind sentence candidates from giraffe treatments."""

    if max_per_trace_per_family <= 0:
        raise ValueError("max_per_trace_per_family must be positive")
    if not tokenizer_id or not tokenizer_revision:
        raise ValueError("pinned tokenizer identity is required")
    candidates: list[PrefilteredSentence] = []
    for row in sorted(rollouts, key=lambda item: str(item.get("run_id", ""))):
        if row.get("task") != "giraffe" or row.get("condition") not in {
            "above_good",
            "below_good",
        }:
            continue
        _validate_source_rollout(row)
        trace_id = str(row.get("run_id", ""))
        reasoning = row.get("reasoning")
        threshold = row.get("threshold")
        raw_text = row.get("raw_text")
        token_streams = row.get("token_streams")
        if not trace_id or not isinstance(reasoning, str) or not isinstance(raw_text, str):
            raise AnchorPipelineError("rollout lacks exact trace text")
        if threshold is None or not isinstance(token_streams, Mapping):
            raise AnchorPipelineError("rollout lacks threshold or original token streams")
        token_map = CompletionTokenMap.from_manifest(
            tokenizer=tokenizer,
            raw_text=raw_text,
            token_streams=token_streams,
            skip_special_tokens=True,
        )
        selected_per_family = {name: 0 for name in _FAMILY_PATTERNS}
        for span in sentence_spans(reasoning):
            family = _lexical_family(span)
            if family is None or selected_per_family[family] >= max_per_trace_per_family:
                continue
            request = build_blinded_request(
                trace_id=trace_id,
                source_text=reasoning,
                sentence_index=span.index,
                threshold_value=threshold,
                include_neighbors=True,
            )
            token_span = token_map.map_reasoning_span(
                span.start,
                span.end,
                expected_text=span.text,
            )
            # Sentence resampling freezes the decoded trace immediately before
            # the anchor and reuses the original completion token prefix.  That
            # intervention is only defined when the sentence starts on an
            # original token boundary.  Filtering here is outcome-blind and
            # also avoids paying two classifiers for an unusable candidate.
            if token_span.leading_envelope_text:
                continue
            candidates.append(
                PrefilteredSentence(
                    trace_id=trace_id,
                    source_reasoning=reasoning,
                    request=request,
                    token_span=token_span,
                    lexical_family=family,
                )
            )
            selected_per_family[family] += 1
    if not candidates:
        raise AnchorPipelineError("outcome-blind prefilter produced no candidates")
    blind_rows = [candidate.blind_dict() for candidate in candidates]
    manifest_hash = stable_hash(
        {
            "version": PREFILTER_VERSION,
            "tokenizer_id": tokenizer_id,
            "tokenizer_revision": tokenizer_revision,
            "max_per_trace_per_family": max_per_trace_per_family,
            "candidates": blind_rows,
        }
    )
    return AnchorPrefilterManifest(
        candidates=tuple(candidates),
        tokenizer_id=tokenizer_id,
        tokenizer_revision=tokenizer_revision,
        max_per_trace_per_family=max_per_trace_per_family,
        manifest_hash=manifest_hash,
    )


def classify_prefiltered_sentences(
    manifest: AnchorPrefilterManifest,
    *,
    callers: tuple[ClassificationCaller, ClassificationCaller],
    provenances: tuple[ModelProvenance, ModelProvenance],
    confidence_threshold: float = 0.80,
) -> LockedClassifications:
    """Lock two-route classifications before any outcome strata can be joined."""

    records: list[ClassifiedCandidateRecord] = []
    token_spans: dict[str, OriginalTokenSpan] = {}
    for candidate in manifest.candidates:
        result = classify_primary(
            candidate.request,
            callers=callers,
            provenances=provenances,
            confidence_threshold=confidence_threshold,
        )
        locked = lock_candidate_record(
            request=candidate.request,
            trace_id=candidate.trace_id,
            source_text=candidate.source_reasoning,
            result=result,
            provenance=provenances,
        )
        records.append(locked)
        token_spans[locked.candidate_id] = candidate.token_span
    payload = {
        "prefilter_manifest_hash": manifest.manifest_hash,
        "records": [record.as_dict() for record in records],
        "token_spans": {
            candidate_id: span.as_dict() for candidate_id, span in sorted(token_spans.items())
        },
    }
    return LockedClassifications(
        records=tuple(records),
        token_spans=token_spans,
        prefilter_manifest_hash=manifest.manifest_hash,
        lock_hash=stable_hash(payload),
    )


def attach_frozen_selection_strata(
    locked: LockedClassifications,
    *,
    rollouts: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Join direction/trajectory strata only after validating the lock hash."""

    if (
        stable_hash(
            {
                "prefilter_manifest_hash": locked.prefilter_manifest_hash,
                "records": [record.as_dict() for record in locked.records],
                "token_spans": {
                    candidate_id: span.as_dict()
                    for candidate_id, span in sorted(locked.token_spans.items())
                },
            }
        )
        != locked.lock_hash
    ):
        raise AnchorPipelineError("classification lock hash mismatch")
    rollout_by_id = {str(row.get("run_id")): row for row in rollouts}
    assert_unique(rollout_by_id.values(), "run_id")
    rows: list[dict[str, Any]] = []
    for record in locked.records:
        if not record.eligible or record.label not in DEFAULT_SENTENCE_CLASSES:
            continue
        source = rollout_by_id.get(record.trace_id)
        if source is None:
            raise AnchorPipelineError(f"classified trace is absent: {record.trace_id}")
        _validate_source_rollout(source)
        first_good = source.get("first_good_side")
        final_flip = source.get("first_to_final_flip")
        if type(first_good) is not bool or type(final_flip) is not bool:
            continue
        token_span = locked.token_spans[record.candidate_id]
        provenance_rows = [item.as_dict() for item in record.model_provenance]
        judgments = [judgment.as_dict() for judgment in record.judgments]
        anchor_provenance = {
            "task": source.get("task"),
            "threshold": source.get("threshold"),
            "prompt_hash": source.get("prompt_hash"),
            "model_hash": source.get("model_hash"),
            "source_rollout_hash": source.get("record_hash"),
            "reasoning_span_hash": stable_hash(record.sentence_text),
            "completion_token_ids_hash": token_span.completion_token_ids_hash,
            "token_span": token_span.as_dict(),
            "classifier_provenance_hash": stable_hash(provenance_rows),
            "classifier_judgments_hash": stable_hash(judgments),
            "classification_candidate_id": record.candidate_id,
            "classification_lock_hash": locked.lock_hash,
            "prefilter_manifest_hash": locked.prefilter_manifest_hash,
        }
        rows.append(
            {
                "trace_id": record.trace_id,
                "sentence_class": record.label,
                "direction": source.get("condition"),
                "sentence_index": record.sentence_index,
                "sentence_text": record.sentence_text,
                "char_start": record.char_start,
                "char_end": record.char_end,
                "initial_side": "good" if first_good else "bad",
                "final_flip": final_flip,
                "eligible": True,
                "classifier_confidence": record.confidence,
                "classifier_resolution": record.resolution,
                "anchor_provenance": anchor_provenance,
                "synthetic_smoke": False,
            }
        )
    if not rows:
        raise AnchorPipelineError("no locked classifications had complete selection strata")
    return rows


__all__ = [
    "PREFILTER_VERSION",
    "AnchorPipelineError",
    "AnchorPrefilterManifest",
    "LockedClassifications",
    "PrefilteredSentence",
    "attach_frozen_selection_strata",
    "classify_prefiltered_sentences",
    "prefilter_anchor_sentences",
]
