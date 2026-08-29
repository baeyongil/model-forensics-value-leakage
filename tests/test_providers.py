from __future__ import annotations

import hashlib
import multiprocessing
import os
import stat
import time
from pathlib import Path

import pytest

from model_forensics.adjudication import (
    FINAL_ANSWER_INSTRUMENT,
    BlindedAdjudicationCase,
    build_adjudication_request,
)
from model_forensics.budget import BudgetExceeded, BudgetLimits, CostEntry, CostLedger
from model_forensics.classification import (
    ACCURACY_COMMITMENT,
    build_blinded_request,
    classify_primary,
)
from model_forensics.io import stable_hash
from model_forensics.paid_response_store import PaidResponseStore, PaidResponseStoreError
from model_forensics.providers import (
    HTTPResult,
    OpenRouterAdjudicationCaller,
    OpenRouterClassificationCaller,
    OpenRouterDispatchGuard,
    OpenRouterJSONClient,
    OpenRouterRequestSpec,
    ProviderError,
    TokenPrice,
    preflight_openrouter_phase,
)


def _concurrent_paid_call_worker(
    ledger_path: str,
    store_path: str,
    transport_marker: str,
    start: object,
    results: object,
) -> None:
    def transport(*_args):
        with Path(transport_marker).open("a", encoding="utf-8") as handle:
            handle.write("transport\n")
        time.sleep(0.15)
        return HTTPResult(
            200,
            {
                "choices": [{"message": {"content": "{}"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "cost": 0.0},
            },
        )

    start.wait()  # type: ignore[attr-defined]
    client = OpenRouterJSONClient(
        model_id="vendor/model",
        price=TokenPrice(1, 1),
        ledger=CostLedger(ledger_path),
        api_key_env="TEST_OPENROUTER_KEY",
        paid_response_store=PaidResponseStore(store_path),
        transport=transport,
    )
    results.put(  # type: ignore[attr-defined]
        client.complete_json(request_id="same-request", user_content="payload", purpose="test")
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


def test_provider_usage_cannot_reduce_cost_below_frozen_token_prices(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TEST_OPENROUTER_KEY", "secret")
    client = OpenRouterJSONClient(
        model_id="vendor/model",
        price=TokenPrice(1_000, 1_000),
        ledger=CostLedger(tmp_path / "cost.yaml"),
        api_key_env="TEST_OPENROUTER_KEY",
        transport=lambda *_args: HTTPResult(
            200,
            {
                "choices": [{"message": {"content": "{}"}}],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "cost": 0.0,
                },
            },
        ),
    )

    assert (
        client.complete_json(
            request_id="reported-undercharge",
            user_content="payload",
            purpose="test",
        )
        == "{}"
    )
    assert client.metadata["computed_cost_usd"] == pytest.approx(0.002)
    assert client.metadata["reported_cost_usd"] == 0.0
    assert client.metadata["charged_cost_usd"] == pytest.approx(0.002)


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


def test_provider_does_not_retry_transient_statuses_automatically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempts = 0

    def transport(*args):
        nonlocal attempts
        attempts += 1
        return HTTPResult(429, {"error": {"type": "rate_limit"}})

    caller = _caller(tmp_path, monkeypatch, transport)
    with pytest.raises(ProviderError, match="HTTP 429"):
        caller.complete(_request())
    assert attempts == 1


def test_ambiguous_timeout_is_marked_and_blocks_automatic_or_resumed_transport(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempts = 0

    def transport(*args):
        nonlocal attempts
        attempts += 1
        raise TimeoutError("sensitive transport internals")

    monkeypatch.setenv("TEST_OPENROUTER_KEY", "super-secret")
    store = PaidResponseStore(tmp_path / "responses")
    ledger = CostLedger(tmp_path / "cost.yaml")
    common = {
        "model_id": "anthropic/test-model",
        "price": TokenPrice(1, 5),
        "ledger": ledger,
        "api_key_env": "TEST_OPENROUTER_KEY",
        "paid_response_store": store,
        "transport": transport,
    }
    first = OpenRouterJSONClient(**common)
    with pytest.raises(ProviderError, match="uncertain paid attempt") as first_error:
        first.complete_json(request_id="ambiguous", user_content="payload", purpose="test")
    assert "sensitive" not in str(first_error.value)
    assert attempts == 1

    resumed = OpenRouterJSONClient(**common)
    with pytest.raises(ProviderError, match="reconciliation"):
        resumed.complete_json(request_id="ambiguous", user_content="payload", purpose="test")
    assert attempts == 1


def test_write_ahead_marker_is_durable_before_transport_and_survives_abrupt_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TEST_OPENROUTER_KEY", "secret")
    store = PaidResponseStore(tmp_path / "responses")
    ledger = CostLedger(tmp_path / "cost.yaml")
    calls = 0

    def abruptly_stopped_transport(*_args):
        nonlocal calls
        calls += 1
        key = store.key(request_id="abrupt", model_id="vendor/model", purpose="test")
        fingerprint = store.fingerprint(
            endpoint="https://openrouter.ai/api/v1/chat/completions",
            model_id="vendor/model",
            purpose="test",
            system_prompt=None,
            user_content="payload",
            decoding={
                "temperature": 0,
                "max_tokens": 512,
                "response_format": "json_object",
                "preflight_input_bound": "one_token_per_utf8_byte_plus_64_per_message",
            },
        )
        marker = store.load_uncertain_attempt(key=key, request_fingerprint=fingerprint)
        assert marker is not None
        assert marker["attempt_state"] == "dispatch_started_outcome_unknown"
        raise KeyboardInterrupt

    first = OpenRouterJSONClient(
        model_id="vendor/model",
        price=TokenPrice(1, 1),
        ledger=ledger,
        api_key_env="TEST_OPENROUTER_KEY",
        paid_response_store=store,
        transport=abruptly_stopped_transport,
    )
    with pytest.raises(KeyboardInterrupt):
        first.complete_json(request_id="abrupt", user_content="payload", purpose="test")
    assert calls == 1

    resumed = OpenRouterJSONClient(
        model_id="vendor/model",
        price=TokenPrice(1, 1),
        ledger=ledger,
        api_key_env="TEST_OPENROUTER_KEY",
        paid_response_store=store,
        transport=lambda *_args: (_ for _ in ()).throw(AssertionError("must block")),
    )
    with pytest.raises(ProviderError, match="reconciliation"):
        resumed.complete_json(request_id="abrupt", user_content="payload", purpose="test")
    assert calls == 1


def test_crash_after_durable_marker_but_before_transport_blocks_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TEST_OPENROUTER_KEY", "secret")
    store = PaidResponseStore(tmp_path / "responses")
    ledger = CostLedger(tmp_path / "cost.yaml")
    real_mark = store.mark_uncertain_attempt
    transport_calls = 0

    def mark_then_crash(**kwargs):
        real_mark(**kwargs)
        raise KeyboardInterrupt

    def transport(*_args):
        nonlocal transport_calls
        transport_calls += 1
        raise AssertionError("transport must not start after the simulated crash")

    monkeypatch.setattr(store, "mark_uncertain_attempt", mark_then_crash)
    first = OpenRouterJSONClient(
        model_id="vendor/model",
        price=TokenPrice(1, 1),
        ledger=ledger,
        api_key_env="TEST_OPENROUTER_KEY",
        paid_response_store=store,
        transport=transport,
    )
    with pytest.raises(KeyboardInterrupt):
        first.complete_json(request_id="pre-transport", user_content="payload", purpose="test")
    assert transport_calls == 0

    resumed = OpenRouterJSONClient(
        model_id="vendor/model",
        price=TokenPrice(1, 1),
        ledger=ledger,
        api_key_env="TEST_OPENROUTER_KEY",
        paid_response_store=PaidResponseStore(store.directory),
        transport=transport,
    )
    with pytest.raises(ProviderError, match="reconciliation"):
        resumed.complete_json(request_id="pre-transport", user_content="payload", purpose="test")
    assert transport_calls == 0


def test_store_path_swap_during_transport_keeps_marker_on_anchored_inode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TEST_OPENROUTER_KEY", "secret")
    store = PaidResponseStore(tmp_path / "responses")
    ledger = CostLedger(tmp_path / "cost.yaml")
    archived = tmp_path / "responses-before-transport-swap"
    transport_calls = 0

    def swap_then_return(*_args):
        nonlocal transport_calls
        transport_calls += 1
        store.directory.rename(archived)
        store.directory.mkdir()
        return HTTPResult(
            200,
            {
                "choices": [{"message": {"content": "{}"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "cost": 0.0},
            },
        )

    first = OpenRouterJSONClient(
        model_id="vendor/model",
        price=TokenPrice(1, 1),
        ledger=ledger,
        api_key_env="TEST_OPENROUTER_KEY",
        paid_response_store=store,
        transport=swap_then_return,
    )
    with pytest.raises(PaidResponseStoreError, match="anchored directory"):
        first.complete_json(request_id="mid-transport-swap", user_content="payload", purpose="test")
    assert transport_calls == 1

    resumed = OpenRouterJSONClient(
        model_id="vendor/model",
        price=TokenPrice(1, 1),
        ledger=ledger,
        api_key_env="TEST_OPENROUTER_KEY",
        paid_response_store=PaidResponseStore(archived),
        transport=lambda *_args: (_ for _ in ()).throw(AssertionError("must block")),
    )
    with pytest.raises(ProviderError, match="reconciliation"):
        resumed.complete_json(
            request_id="mid-transport-swap", user_content="payload", purpose="test"
        )
    assert transport_calls == 1


def test_store_root_is_fsynced_when_control_directories_are_first_created(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fsynced_directories: list[tuple[int, int]] = []
    real_fsync = os.fsync

    def recording_fsync(descriptor: int) -> None:
        observed = os.fstat(descriptor)
        if stat.S_ISDIR(observed.st_mode):
            fsynced_directories.append((int(observed.st_dev), int(observed.st_ino)))
        real_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", recording_fsync)
    store = PaidResponseStore(tmp_path / "responses")
    root = store.directory.stat()
    root_identity = (int(root.st_dev), int(root.st_ino))

    assert (store.directory / ".claims").is_dir()
    assert (store.directory / ".uncertain").is_dir()
    assert fsynced_directories.count(root_identity) >= 2


def test_write_ahead_marker_remains_when_checkpoint_commit_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TEST_OPENROUTER_KEY", "secret")
    store = PaidResponseStore(tmp_path / "responses")
    ledger = CostLedger(tmp_path / "cost.yaml")
    transport_calls = 0

    def transport(*_args):
        nonlocal transport_calls
        transport_calls += 1
        return HTTPResult(
            200,
            {
                "choices": [{"message": {"content": "{}"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )

    monkeypatch.setattr(
        store,
        "commit",
        lambda **_kwargs: (_ for _ in ()).throw(
            PaidResponseStoreError("simulated durable commit failure")
        ),
    )
    client = OpenRouterJSONClient(
        model_id="vendor/model",
        price=TokenPrice(1, 1),
        ledger=ledger,
        api_key_env="TEST_OPENROUTER_KEY",
        paid_response_store=store,
        transport=transport,
    )
    with pytest.raises(PaidResponseStoreError, match="commit failure"):
        client.complete_json(request_id="commit-fail", user_content="payload", purpose="test")
    assert transport_calls == 1

    resumed = OpenRouterJSONClient(
        model_id="vendor/model",
        price=TokenPrice(1, 1),
        ledger=ledger,
        api_key_env="TEST_OPENROUTER_KEY",
        paid_response_store=PaidResponseStore(store.directory),
        transport=lambda *_args: (_ for _ in ()).throw(AssertionError("must block")),
    )
    with pytest.raises(ProviderError, match="reconciliation"):
        resumed.complete_json(request_id="commit-fail", user_content="payload", purpose="test")
    assert transport_calls == 1


def test_checkpoint_and_marker_reconcile_after_crash_before_marker_resolution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TEST_OPENROUTER_KEY", "secret")
    store = PaidResponseStore(tmp_path / "responses")
    ledger = CostLedger(tmp_path / "cost.yaml")
    real_commit = store.commit
    transport_calls = 0

    def transport(*_args):
        nonlocal transport_calls
        transport_calls += 1
        return HTTPResult(
            200,
            {
                "choices": [{"message": {"content": "{}"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "cost": 0.0},
            },
        )

    def commit_then_crash(**kwargs):
        real_commit(**kwargs)
        raise KeyboardInterrupt

    monkeypatch.setattr(store, "commit", commit_then_crash)
    first = OpenRouterJSONClient(
        model_id="vendor/model",
        price=TokenPrice(1, 1),
        ledger=ledger,
        api_key_env="TEST_OPENROUTER_KEY",
        paid_response_store=store,
        transport=transport,
    )
    with pytest.raises(KeyboardInterrupt):
        first.complete_json(
            request_id="commit-before-resolve",
            user_content="payload",
            purpose="test",
        )
    assert transport_calls == 1

    resumed = OpenRouterJSONClient(
        model_id="vendor/model",
        price=TokenPrice(1, 1),
        ledger=ledger,
        api_key_env="TEST_OPENROUTER_KEY",
        paid_response_store=PaidResponseStore(store.directory),
        transport=lambda *_args: (_ for _ in ()).throw(
            AssertionError("durable checkpoint must replay")
        ),
    )
    assert (
        resumed.complete_json(
            request_id="commit-before-resolve",
            user_content="payload",
            purpose="test",
        )
        == "{}"
    )
    assert transport_calls == 1


def test_committed_response_replays_and_settles_after_interrupted_settlement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TEST_OPENROUTER_KEY", "secret")
    store = PaidResponseStore(tmp_path / "responses")
    ledger = CostLedger(tmp_path / "cost.yaml")
    real_settle = ledger.settle_reservation
    settlement_calls = 0
    transport_calls = 0

    def interrupted_settlement(*args, **kwargs):
        nonlocal settlement_calls
        settlement_calls += 1
        if settlement_calls == 1:
            raise RuntimeError("simulated crash after checkpoint commit")
        return real_settle(*args, **kwargs)

    def transport(*_args):
        nonlocal transport_calls
        transport_calls += 1
        return HTTPResult(
            200,
            {
                "choices": [{"message": {"content": "{}"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "cost": 0.0},
            },
        )

    monkeypatch.setattr(ledger, "settle_reservation", interrupted_settlement)
    first = OpenRouterJSONClient(
        model_id="vendor/model",
        price=TokenPrice(1, 1),
        ledger=ledger,
        api_key_env="TEST_OPENROUTER_KEY",
        paid_response_store=store,
        transport=transport,
    )
    with pytest.raises(RuntimeError, match="after checkpoint commit"):
        first.complete_json(request_id="settle", user_content="payload", purpose="test")
    assert transport_calls == 1

    second = OpenRouterJSONClient(
        model_id="vendor/model",
        price=TokenPrice(1, 1),
        ledger=ledger,
        api_key_env="TEST_OPENROUTER_KEY",
        paid_response_store=store,
        transport=lambda *_args: (_ for _ in ()).throw(AssertionError("must replay")),
    )
    assert second.complete_json(request_id="settle", user_content="payload", purpose="test") == "{}"
    assert transport_calls == 1
    assert settlement_calls == 2


def test_cached_replay_succeeds_with_zero_budget_headroom(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TEST_OPENROUTER_KEY", "secret")
    price = TokenPrice(1_000_000, 1_000_000)
    store = PaidResponseStore(tmp_path / "responses")
    ledger = CostLedger(tmp_path / "cost.yaml", BudgetLimits(gpu=220, api=1.0, total=1.0))
    spec = OpenRouterRequestSpec(
        route="test",
        model_id="vendor/model",
        price=price,
        request_id="cached",
        purpose="test",
        user_content="payload",
        max_output_tokens=1,
        paid_response_store=store,
    )
    # Use a smaller exact reservation to consume the entire cap, then settle it.
    ledger.reserve(
        spec.reservation_id,
        CostEntry(
            kind="api",
            amount_usd=1.0,
            description=f"preflight OpenRouter test {stable_hash('cached')}",
            status="estimated",
        ),
    )
    store.commit(
        key=spec.store_key,
        request_fingerprint=spec.request_fingerprint,
        logical_request_hash=stable_hash(spec.request_id),
        model_id=spec.model_id,
        purpose=spec.purpose,
        http_status=200,
        response_body={
            "usage": {"prompt_tokens": 0, "completion_tokens": 1, "cost": 1.0},
            "choices": [{"message": {"content": "{}"}}],
        },
    )
    ledger.settle_reservation(
        spec.reservation_id,
        CostEntry(
            kind="api", amount_usd=1.0, description=f"OpenRouter test {stable_hash('cached')}"
        ),
    )
    preflight = preflight_openrouter_phase(phase="positions_api", requests=(spec,), ledger=ledger)
    client = OpenRouterJSONClient(
        model_id=spec.model_id,
        price=spec.price,
        ledger=ledger,
        max_output_tokens=spec.max_output_tokens,
        paid_response_store=store,
        api_key_env="TEST_OPENROUTER_KEY",
        dispatch_guard=OpenRouterDispatchGuard(preflight),
        dispatch_route=spec.route,
        transport=lambda *_args: (_ for _ in ()).throw(AssertionError("no transport")),
    )
    assert (
        client.complete_json(
            request_id=spec.request_id, user_content=spec.user_content, purpose=spec.purpose
        )
        == "{}"
    )


def test_input_bound_counts_utf8_bytes_not_unicode_code_points(tmp_path: Path) -> None:
    spec = OpenRouterRequestSpec(
        route="test",
        model_id="vendor/model",
        price=TokenPrice(1, 1),
        request_id="unicode",
        purpose="test",
        user_content="🙂한글",
        paid_response_store=PaidResponseStore(tmp_path / "responses"),
    )
    assert spec.input_token_upper_bound == len("🙂한글".encode()) + 64


def test_cross_process_request_claim_allows_only_one_paid_transport(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TEST_OPENROUTER_KEY", "secret")
    context = multiprocessing.get_context("fork")
    start = context.Event()
    results = context.Queue()
    marker = tmp_path / "transports.txt"
    arguments = (
        str(tmp_path / "cost.yaml"),
        str(tmp_path / "responses"),
        str(marker),
        start,
        results,
    )
    workers = [
        context.Process(target=_concurrent_paid_call_worker, args=arguments) for _ in range(2)
    ]
    for worker in workers:
        worker.start()
    start.set()
    for worker in workers:
        worker.join(timeout=5)
        assert worker.exitcode == 0
    assert [results.get(timeout=1), results.get(timeout=1)] == ["{}", "{}"]
    assert marker.read_text(encoding="utf-8").splitlines() == ["transport"]


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
    assert client.complete_json(request_id="unit-1", user_content="first", purpose="final") == "{}"
    with pytest.raises(PaidResponseStoreError, match="fingerprint"):
        client.complete_json(request_id="unit-1", user_content="changed", purpose="final")
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
