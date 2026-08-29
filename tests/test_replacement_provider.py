from __future__ import annotations

from typing import Any

import pytest

from model_forensics.io import stable_hash
from model_forensics.replacement_provider import (
    TwoRouteOpenRouterReplacementClassifier,
    parse_replacement_judgment,
)
from model_forensics.resample_runner import (
    ReplacementClassificationError,
    ReplacementClassificationRequest,
)


class Client:
    def __init__(self, model_id: str, response: str) -> None:
        self.model_id = model_id
        self.model_revision = "frozen"
        self.decoding = {"temperature": 0}
        self.pricing = {"input_per_million": 1.0, "output_per_million": 2.0}
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def complete_json(self, **kwargs: Any) -> str:
        self.calls.append(kwargs)
        return self.response


def _request() -> ReplacementClassificationRequest:
    return ReplacementClassificationRequest(
        request_id="opaque",
        blinded_original_sentence="I must remain objective.",
        blinded_replacement_sentence="Next I multiply the two task factors.",
        target_sentence_class="accuracy_commitment",
        neutral_control_id="neutral-continuation",
        neutral_control_function="A coherent task-relevant continuation without the target feature.",
        neutral_control_version="v1",
    )


def _response(feature: bool = True, matched: bool = True, confidence: float = 0.95) -> str:
    return (
        "{"
        f'"target_feature_absent_or_changed":{str(feature).lower()},'
        f'"neutral_control_function_matched":{str(matched).lower()},'
        f'"confidence":{confidence},"rationale":"visible replacement changes function"'
        "}"
    )


def test_two_routes_must_agree_and_receive_only_visible_payload() -> None:
    first = Client("judge-a", _response())
    second = Client("judge-b", _response())
    classifier = TwoRouteOpenRouterReplacementClassifier((first, second))
    result = classifier.classify(_request())
    assert result.adjudication_valid is True
    assert result.target_feature_absent_or_changed is True
    assert result.neutral_control_function_matched is True
    assert result.classifier_provenance_hash == stable_hash(dict(classifier.provenance))
    assert len(set(result.raw_judgment_hashes)) == 2
    for client in (first, second):
        content = client.calls[0]["user_content"]
        assert "condition" not in content
        assert "threshold" not in content
        assert "outcome" not in content


def test_disagreement_is_invalid_instead_of_resolved_post_hoc() -> None:
    classifier = TwoRouteOpenRouterReplacementClassifier(
        (Client("judge-a", _response()), Client("judge-b", _response(feature=False)))
    )
    result = classifier.classify(_request())
    assert result.adjudication_valid is False
    assert result.target_feature_absent_or_changed is None
    assert result.neutral_control_function_matched is None


@pytest.mark.parametrize(
    "raw",
    [
        "not json",
        '{"target_feature_absent_or_changed":true}',
        (
            '{"target_feature_absent_or_changed":true,'
            '"neutral_control_function_matched":true,"confidence":true,"rationale":"x"}'
        ),
    ],
)
def test_strict_replacement_schema(raw: str) -> None:
    with pytest.raises(ReplacementClassificationError):
        parse_replacement_judgment(raw)
