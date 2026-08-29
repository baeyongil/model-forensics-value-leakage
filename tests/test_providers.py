from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from model_forensics.adjudication import (
    FINAL_ANSWER_INSTRUMENT,
    BlindedAdjudicationCase,
    build_adjudication_request,
)
from model_forensics.budget import BudgetExceeded, BudgetLimits, CostLedger
from model_forensics.classification import (
    ACCURACY_COMMITMENT,
    build_blinded_request,
    classify_primary,
)
from model_forensics.paid_response_store import PaidResponseStore, PaidResponseStoreError
from model_forensics.providers import (
    HTTPResult,
    OpenRouterAdjudicationCaller,
    OpenRouterClassificationCaller,
    OpenRouterJSONClient,
    ProviderError,
    TokenPrice,
)


def _request():
    case = BlindedAdjudicationCase("How many?", "Estimate 42.", "42")
    return build_adjudication_request(case, FINAL_ANSWER_INSTRUMENT)


def _caller(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    transport,
    *,
    limit: float = 1.0,
) -> OpenRouterAdjudicationCaller:
    monkeypatch.setenv("TEST_OPENROUTER_KEY", "super-secret")
    ledger = CostLedger(
        tmp_path / "cost.yaml",
        BudgetLimits(gpu=220, api=limit, total=250),
    )
    return OpenRouterAdjudicationCaller(
        model_id="anthropic/test-model",
        price=TokenPrice(input_per_million=1, output_per_million=5),
        ledger=ledger,
        api_key_env="TEST_OPENROUTER_KEY",
        transport=transport,
        sleep=lambda _: None,
    )


def test_openrouter_caller_is_blind_secret_safe_and_cost_accounted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed = {}

    def transport(url, headers, payload, timeout):
        observed.update(url=url, headers=headers, payload=payload, timeout=timeout)
        return HTTPResult(
            200,
            {
                "id": "response-123",
                "choices": [{"message": {"content": '{"status":"KNOWN","value":"42"}'}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20, "cost": 0.0002},
            },
        )

    caller = _caller(tmp_path, monkeypatch, transport)
    assert caller.complete(_request()) == '{"status":"KNOWN","value":"42"}'
    assert set(observed["payload"]["messages"][1]) == {"role", "content"}
    user_content = observed["payload"]["messages"][1]["content"]
    assert "condition" not in user_content and "threshold" not in user_content
    assert "super-secret" not in repr(caller.provenance.to_dict())
    assert caller.provenance.metadata["charged_cost_usd"] == pytest.approx(0.0002)
    ledger_text = (tmp_path / "cost.yaml").read_text(encoding="utf-8")
    assert "super-secret" not in ledger_text
    assert "amount_usd: 0.0002" in ledger_text


def test_preflight_fails_before_transport_when_budget_cannot_fit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    called = False

    def transport(*args):
        nonlocal called
        called = True
        raise AssertionError("transport must not run")

    caller = _caller(tmp_path, monkeypatch, transport, limit=0.000001)
    with pytest.raises(BudgetExceeded):
        caller.complete(_request())
    assert called is False


def test_provider_retries_only_transient_statuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    statuses = iter((429, 200))

    def transport(*args):
        status = next(statuses)
        if status == 429:
            return HTTPResult(status, {"error": {"type": "rate_limit"}})
        return HTTPResult(
            status,
            {
                "choices": [{"message": {"content": "{}"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )

    caller = _caller(tmp_path, monkeypatch, transport)
    assert caller.complete(_request()) == "{}"


def test_provider_retries_transient_transport_failure_without_leaking_details(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempts = 0

    def transport(*args):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise TimeoutError("sensitive transport internals")
        return HTTPResult(
            200,
            {
                "choices": [{"message": {"content": "{}"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )

    caller = _caller(tmp_path, monkeypatch, transport)
    assert caller.complete(_request()) == "{}"
    assert attempts == 2
    assert caller.provenance.metadata["attempts_used"] == 2


def test_provider_rejects_success_without_usage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    caller = _caller(
        tmp_path,
        monkeypatch,
        lambda *args: HTTPResult(
            200,
            {"choices": [{"message": {"content": "{}"}}]},
        ),
    )
    with pytest.raises(ProviderError, match="usage"):
        caller.complete(_request())


def test_missing_secret_fails_without_echoing_any_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("MISSING_TEST_KEY", raising=False)
    with pytest.raises(RuntimeError, match="MISSING_TEST_KEY"):
        OpenRouterAdjudicationCaller(
            model_id="m",
            price=TokenPrice(1, 1),
            ledger=CostLedger(tmp_path / "cost.yaml"),
            api_key_env="MISSING_TEST_KEY",
        )


def test_generic_json_client_uses_one_user_message_and_accounts_invalid_paid_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TEST_OPENROUTER_KEY", "generic-super-secret")
    observed = {}

    def transport(url, headers, payload, timeout):
        observed.update(url=url, headers=headers, payload=payload, timeout=timeout)
        return HTTPResult(
            200,
            {
                "id": "paid-but-malformed",
                "choices": [{"message": {"content": "not-json"}}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 3, "cost": 0.001},
            },
        )

    ledger = CostLedger(tmp_path / "generic-cost.yaml")
    client = OpenRouterJSONClient(
        model_id="example/generic-json",
        price=TokenPrice(1, 2),
        ledger=ledger,
        api_key_env="TEST_OPENROUTER_KEY",
        transport=transport,
        sleep=lambda _: None,
    )
    with pytest.raises(ProviderError, match="strict JSON"):
        client.complete_json(
            request_id="opaque-request",
            user_content="Return a JSON object.",
            purpose="test",
        )

    assert observed["payload"]["messages"] == [{"role": "user", "content": "Return a JSON object."}]
    assert observed["payload"]["response_format"] == {"type": "json_object"}
    assert client.metadata["charged_cost_usd"] == pytest.approx(0.001)
    assert client.metadata["logical_request_hash"].startswith("sha256:")
    assert "opaque-request" not in (tmp_path / "generic-cost.yaml").read_text(encoding="utf-8")
    assert "generic-super-secret" not in repr(client.provenance)


def test_paid_response_is_replayed_without_second_charge_even_when_json_is_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TEST_OPENROUTER_KEY", "checkpoint-secret")
    calls = 0

    def transport(*args):
        nonlocal calls
        calls += 1
        return HTTPResult(
            200,
            {
                "id": "paid-invalid",
                "choices": [{"message": {"content": "not-json"}}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 3, "cost": 0.001},
            },
        )

    ledger = CostLedger(tmp_path / "cost.yaml")
    store = PaidResponseStore(tmp_path / "responses")
    common = {
        "model_id": "vendor/model",
        "price": TokenPrice(1, 2),
        "ledger": ledger,
        "api_key_env": "TEST_OPENROUTER_KEY",
        "paid_response_store": store,
        "sleep": lambda _: None,
    }
    first = OpenRouterJSONClient(transport=transport, **common)
    with pytest.raises(ProviderError, match="strict JSON"):
        first.complete_json(
            request_id="unit-1",
            user_content="Return JSON.",
            purpose="final",
        )
    assert calls == 1
    assert len(list((tmp_path / "responses").glob("*.json"))) == 1

    second = OpenRouterJSONClient(
        transport=lambda *args: (_ for _ in ()).throw(
            AssertionError("checkpoint replay must not call transport")
        ),
        **common,
    )
    with pytest.raises(ProviderError, match="strict JSON"):
        second.complete_json(
            request_id="unit-1",
            user_content="Return JSON.",
            purpose="final",
        )
    assert calls == 1
    assert second.metadata["replayed_from_checkpoint"] is True
    assert len(ledger.document()["entries"]) == 1


def test_paid_response_checkpoint_rejects_request_drift_before_transport(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TEST_OPENROUTER_KEY", "checkpoint-secret")
    calls = 0

    def transport(*args):
        nonlocal calls
        calls += 1
        return HTTPResult(
            200,
            {
                "choices": [{"message": {"content": "{}"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )

    client = OpenRouterJSONClient(
        model_id="vendor/model",
        price=TokenPrice(1, 2),
        ledger=CostLedger(tmp_path / "cost.yaml"),
        api_key_env="TEST_OPENROUTER_KEY",
        transport=transport,
        paid_response_store=PaidResponseStore(tmp_path / "responses"),
    )
    assert (
        client.complete_json(
            request_id="unit-1", user_content="first", purpose="final"
        )
        == "{}"
    )
    with pytest.raises(PaidResponseStoreError, match="fingerprint"):
        client.complete_json(
            request_id="unit-1", user_content="changed", purpose="final"
        )
    assert calls == 1


def test_openrouter_classification_callers_plug_into_primary_anchor_classifier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TEST_OPENROUTER_KEY", "classification-super-secret")
    request = build_blinded_request(
        trace_id="hidden-above-good-outcome",
        source_text="I should remain accurate and objective.",
        sentence_index=0,
        threshold_value=41_000_000,
        include_neighbors=False,
    )
    observed_payloads: list[dict] = []

    def transport(*args):
        payload = args[2]
        observed_payloads.append(payload)
        return HTTPResult(
            200,
            {
                "id": f"classification-{len(observed_payloads)}",
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"label":"accuracy_commitment","confidence":0.94,'
                                '"rationale":"explicit accuracy commitment"}'
                            )
                        }
                    }
                ],
                "usage": {"prompt_tokens": 200, "completion_tokens": 25},
            },
        )

    ledger = CostLedger(tmp_path / "classification-cost.yaml")
    common = {
        "price": TokenPrice(1, 2),
        "ledger": ledger,
        "api_key_env": "TEST_OPENROUTER_KEY",
        "transport": transport,
        "sleep": lambda _: None,
    }
    first = OpenRouterClassificationCaller(model_id="vendor/judge-a", **common)
    second = OpenRouterClassificationCaller(model_id="vendor/judge-b", **common)
    result = classify_primary(
        request,
        callers=(first, second),
        provenances=(first.provenance, second.provenance),
    )

    assert result.eligible is True
    assert result.label == ACCURACY_COMMITMENT
    assert len(observed_payloads) == 2
    assert all(
        payload["messages"] == [{"role": "user", "content": request.prompt}]
        for payload in observed_payloads
    )
    assert all(payload["temperature"] == 0 for payload in observed_payloads)
    assert first.usage_metadata["prompt_hash"] == request.prompt_hash
    assert first.usage_metadata["input_hash"] == request.input_hash
    assert first.usage_metadata["charged_cost_usd"] > 0
    assert len(first.usage_records) == 1
    assert first.usage_records[0]["logical_request_hash"].startswith("sha256:")
    audit_text = repr(first.provenance.as_dict()) + repr(first.usage_metadata)
    assert "classification-super-secret" not in audit_text


def test_classification_wrapper_rejects_prompt_hash_mismatch_before_transport(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TEST_OPENROUTER_KEY", "secret")
    called = False

    def transport(*args):
        nonlocal called
        called = True
        raise AssertionError("transport must not run")

    caller = OpenRouterClassificationCaller(
        model_id="vendor/judge",
        price=TokenPrice(1, 1),
        ledger=CostLedger(tmp_path / "cost.yaml"),
        api_key_env="TEST_OPENROUTER_KEY",
        transport=transport,
    )
    with pytest.raises(ProviderError, match="prompt hash mismatch"):
        caller(
            prompt="frozen prompt",
            judgment_id="a" * 64,
            input_hash="b" * 64,
            prompt_hash="c" * 64,
        )
    prompt = "Frozen classifier.\nBlinded input:\n{}"
    with pytest.raises(ProviderError, match="blinded-input hash mismatch"):
        caller(
            prompt=prompt,
            judgment_id="a" * 64,
            input_hash="b" * 64,
            prompt_hash=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        )
    assert called is False
