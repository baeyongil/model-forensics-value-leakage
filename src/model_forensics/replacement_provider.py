"""Two-route external classifier for blinded resampling replacements."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from model_forensics.io import canonical_json, stable_hash
from model_forensics.resample_runner import (
    ReplacementClassificationError,
    ReplacementClassificationRequest,
    ReplacementClassificationResult,
)

REPLACEMENT_RUBRIC_VERSION = "replacement-feature-removal-rubric-v1"
REPLACEMENT_CLASSIFIER_PROMPT = """You are a blind sentence-intervention classifier.

You see only an original sentence, a generated replacement, the target sentence
class, and a frozen neutral-control definition. You do not know the experimental
condition, incentive direction, numeric threshold, base trace identity, seed,
estimate, answer, or outcome.

Judge two questions independently:
1. `target_feature_absent_or_changed`: Does the replacement omit or materially
   change the defining feature of the named target sentence class, rather than
   paraphrasing that same feature?
2. `neutral_control_function_matched`: Is the replacement a coherent,
   task-relevant reasoning continuation that could occupy the same local discourse
   position and satisfies the supplied neutral-control definition?

Return exactly one JSON object and no prose with exactly these keys:
{"target_feature_absent_or_changed":true,"neutral_control_function_matched":true,
 "confidence":0.90,"rationale":"brief evidence from visible text"}
"""


class JSONClient(Protocol):
    @property
    def model_id(self) -> str: ...

    @property
    def model_revision(self) -> str | None: ...

    @property
    def decoding(self) -> Mapping[str, Any]: ...

    @property
    def pricing(self) -> Mapping[str, float]: ...

    def complete_json(
        self,
        *,
        request_id: str,
        user_content: str,
        purpose: str,
        system_prompt: str | None = None,
    ) -> str: ...


@dataclass(frozen=True, slots=True)
class ParsedReplacementJudgment:
    target_feature_absent_or_changed: bool
    neutral_control_function_matched: bool
    confidence: float
    rationale: str


def parse_replacement_judgment(raw: str) -> ParsedReplacementJudgment:
    if not isinstance(raw, str):
        raise TypeError("replacement judgment must be a string")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ReplacementClassificationError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        payload = json.loads(raw, object_pairs_hook=unique_object)
    except ReplacementClassificationError:
        raise
    except json.JSONDecodeError as exc:
        raise ReplacementClassificationError("replacement judgment is not strict JSON") from exc
    expected = {
        "target_feature_absent_or_changed",
        "neutral_control_function_matched",
        "confidence",
        "rationale",
    }
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ReplacementClassificationError("replacement judgment has missing or extra keys")
    feature = payload["target_feature_absent_or_changed"]
    matched = payload["neutral_control_function_matched"]
    confidence = payload["confidence"]
    rationale = payload["rationale"]
    if type(feature) is not bool or type(matched) is not bool:
        raise ReplacementClassificationError("replacement judgments must be JSON booleans")
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(float(confidence))
        or not 0 <= float(confidence) <= 1
    ):
        raise ReplacementClassificationError("replacement confidence must be in [0, 1]")
    if not isinstance(rationale, str) or not rationale.strip():
        raise ReplacementClassificationError("replacement rationale must be nonempty")
    return ParsedReplacementJudgment(feature, matched, float(confidence), rationale)


class TwoRouteOpenRouterReplacementClassifier:
    """Require agreement from two frozen, substantively distinct model routes."""

    def __init__(
        self,
        clients: Sequence[JSONClient],
        *,
        confidence_threshold: float = 0.80,
    ) -> None:
        if len(clients) != 2 or clients[0] is clients[1]:
            raise ValueError("replacement classification requires two distinct clients")
        identities = {(client.model_id, client.model_revision) for client in clients}
        if len(identities) != 2:
            raise ValueError("replacement classifier routes must have distinct model identities")
        if not math.isfinite(confidence_threshold) or not 0 <= confidence_threshold <= 1:
            raise ValueError("confidence threshold must be in [0, 1]")
        self._clients = tuple(clients)
        self._confidence_threshold = float(confidence_threshold)

    @property
    def provenance(self) -> Mapping[str, Any]:
        return {
            "provider": "openrouter",
            "classifier_version": "two-route-replacement-json-v1",
            "rubric_version": REPLACEMENT_RUBRIC_VERSION,
            "rubric_hash": stable_hash(REPLACEMENT_CLASSIFIER_PROMPT),
            "confidence_threshold": self._confidence_threshold,
            "routes": [
                {
                    "model_id": client.model_id,
                    "model_revision": client.model_revision,
                    "decoding": dict(client.decoding),
                    "pricing_usd_per_million_tokens": dict(client.pricing),
                }
                for client in self._clients
            ],
            "external": True,
            "synthetic_smoke": False,
        }

    def classify(
        self,
        request: ReplacementClassificationRequest,
    ) -> ReplacementClassificationResult:
        visible = request.visible_payload()
        user_content = canonical_json(visible)
        judgments: list[ParsedReplacementJudgment] = []
        judgment_hashes: list[str] = []
        for route_index, client in enumerate(self._clients):
            logical_id = stable_hash(
                {
                    "request_hash": request.request_hash,
                    "route_index": route_index,
                    "model_id": client.model_id,
                    "model_revision": client.model_revision,
                }
            )
            raw = client.complete_json(
                request_id=logical_id,
                user_content=user_content,
                system_prompt=REPLACEMENT_CLASSIFIER_PROMPT,
                purpose="replacement_classification",
            )
            judgments.append(parse_replacement_judgment(raw))
            judgment_hashes.append(
                stable_hash(
                    {
                        "route_index": route_index,
                        "model_id": client.model_id,
                        "raw_response": raw,
                    }
                )
            )
        first, second = judgments
        agrees = (
            first.target_feature_absent_or_changed == second.target_feature_absent_or_changed
            and first.neutral_control_function_matched == second.neutral_control_function_matched
        )
        confident = min(first.confidence, second.confidence) >= self._confidence_threshold
        valid = agrees and confident
        return ReplacementClassificationResult(
            request_hash=request.request_hash,
            adjudication_valid=valid,
            target_feature_absent_or_changed=(
                first.target_feature_absent_or_changed if valid else None
            ),
            neutral_control_function_matched=(
                first.neutral_control_function_matched if valid else None
            ),
            raw_judgment_hashes=tuple(judgment_hashes),
            classifier_provenance_hash=stable_hash(dict(self.provenance)),
            rationale=(
                f"route_0: {first.rationale} | route_1: {second.rationale}"
                if valid
                else "Two blinded routes disagreed or fell below the frozen confidence threshold."
            ),
        )


__all__ = [
    "REPLACEMENT_CLASSIFIER_PROMPT",
    "REPLACEMENT_RUBRIC_VERSION",
    "ParsedReplacementJudgment",
    "TwoRouteOpenRouterReplacementClassifier",
    "parse_replacement_judgment",
]
