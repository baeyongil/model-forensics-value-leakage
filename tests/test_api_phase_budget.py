from __future__ import annotations

import json
from pathlib import Path

import pytest

from model_forensics import cli
from model_forensics.budget import BudgetExceeded, BudgetLimits, CostEntry, CostLedger
from model_forensics.io import stable_hash, write_json
from model_forensics.paid_phase_receipt import PaidPhaseReceiptStore
from model_forensics.paid_response_store import PaidResponseStore, PaidResponseStoreError
from model_forensics.providers import (
    HTTPResult,
    OpenRouterDispatchGuard,
    OpenRouterJSONClient,
    OpenRouterRequestSpec,
    ProviderError,
    TokenPrice,
    preflight_openrouter_phase,
)


def _ledger(path: Path, *, api: float = 100.0, total: float = 325.0) -> CostLedger:
    return CostLedger(path, BudgetLimits(gpu=220.0, api=api, total=total))


def _spec(
    tmp_path: Path,
    *,
    store: PaidResponseStore | None = None,
    price: TokenPrice | None = None,
    user_content: str = "payload",
) -> OpenRouterRequestSpec:
    return OpenRouterRequestSpec(
        route="primary",
        model_id="provider/model",
        model_revision=None,
        price=price or TokenPrice(1.0, 1.0),
        request_id="request-1",
        purpose="adjudication",
        system_prompt="instrument",
        user_content=user_content,
        max_output_tokens=32,
        paid_response_store=store or PaidResponseStore(tmp_path / "responses"),
    )


def test_whole_phase_over_cap_refuses_before_any_transport(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret")
    transports: list[object] = []

    def transport(*args):  # type: ignore[no-untyped-def]
        transports.append(args)
        raise AssertionError("transport must not run")

    spec = _spec(tmp_path, price=TokenPrice(1_000_000.0, 1_000_000.0))
    client = OpenRouterJSONClient(
        model_id=spec.model_id,
        price=spec.price,
        ledger=_ledger(tmp_path / "cost.yaml", api=1.0, total=1.0),
        max_output_tokens=spec.max_output_tokens,
        paid_response_store=spec.paid_response_store,
        transport=transport,
    )
    with pytest.raises(BudgetExceeded, match="whole API phase"):
        preflight_openrouter_phase(
            phase="behavior_baseline_api",
            requests=(spec,),
            ledger=client._ledger,
        )
    assert transports == []


def test_reservation_identity_is_scoped_to_the_anchored_response_store(
    tmp_path: Path,
) -> None:
    first = _spec(tmp_path, store=PaidResponseStore(tmp_path / "responses-a"))
    second = OpenRouterRequestSpec(
        route="secondary",
        model_id=first.model_id,
        model_revision=first.model_revision,
        price=first.price,
        request_id=first.request_id,
        purpose=first.purpose,
        system_prompt=first.system_prompt,
        user_content=first.user_content,
        max_output_tokens=first.max_output_tokens,
        paid_response_store=PaidResponseStore(tmp_path / "responses-b"),
    )
    ledger_path = tmp_path / "cost.yaml"

    assert first.reservation_id != second.reservation_id
    result = preflight_openrouter_phase(
        phase="positions_api",
        requests=(first, second),
        ledger=_ledger(ledger_path),
    )
    assert result.manifest["additional_commitment_required_usd"] == round(
        first.conservative_cost_usd + second.conservative_cost_usd,
        6,
    )


def test_authenticated_cached_call_is_excluded_from_completion_bound(tmp_path: Path) -> None:
    store = PaidResponseStore(tmp_path / "responses")
    spec = _spec(tmp_path, store=store)
    body = {
        "id": "response",
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "cost": 0.01},
        "choices": [{"message": {"content": '{"status":"UNKNOWN"}'}}],
    }
    ledger = _ledger(tmp_path / "cost.yaml")
    ledger.reserve(
        spec.reservation_id,
        CostEntry(
            kind="api",
            amount_usd=spec.conservative_cost_usd,
            description=f"preflight OpenRouter {spec.purpose} {stable_hash(spec.request_id)}",
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
        response_body=body,
    )
    result = preflight_openrouter_phase(
        phase="positions_api",
        requests=(spec,),
        ledger=ledger,
    )
    assert result.manifest["authenticated_cached_count"] == 1
    assert result.manifest["pending_request_count"] == 0
    assert result.manifest["conservative_pending_usd"] == 0
    assert result.manifest["additional_commitment_required_usd"] == 0

    changed_payload = _spec(
        tmp_path,
        store=store,
        user_content="a different exact payload",
    )
    with pytest.raises(PaidResponseStoreError, match="fingerprint mismatch"):
        preflight_openrouter_phase(
            phase="positions_api",
            requests=(changed_payload,),
            ledger=_ledger(tmp_path / "other-cost.yaml"),
        )


def test_tampered_checkpoint_and_inventory_hash_fail_closed(tmp_path: Path) -> None:
    store = PaidResponseStore(tmp_path / "responses")
    spec = _spec(tmp_path, store=store)
    ledger = _ledger(tmp_path / "cost.yaml")
    ledger.reserve(
        spec.reservation_id,
        CostEntry(
            kind="api",
            amount_usd=spec.conservative_cost_usd,
            description=f"preflight OpenRouter {spec.purpose} {stable_hash(spec.request_id)}",
            status="estimated",
        ),
    )
    committed = store.commit(
        key=spec.store_key,
        request_fingerprint=spec.request_fingerprint,
        logical_request_hash=stable_hash(spec.request_id),
        model_id=spec.model_id,
        purpose=spec.purpose,
        http_status=200,
        response_body={
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            "choices": [{"message": {"content": "{}"}}],
        },
    )
    checkpoint = store.directory / f"{spec.store_key.split(':', 1)[1]}.json"
    tampered = json.loads(checkpoint.read_text(encoding="utf-8"))
    tampered["response_body"]["usage"]["prompt_tokens"] = 2
    write_json(checkpoint, tampered)
    with pytest.raises(PaidResponseStoreError, match="body hash"):
        preflight_openrouter_phase(
            phase="positions_api",
            requests=(spec,),
            ledger=ledger,
        )

    write_json(checkpoint, committed)
    result = preflight_openrouter_phase(
        phase="positions_api",
        requests=(spec,),
        ledger=ledger,
    )
    modified = dict(result.manifest)
    modified["pending_request_count"] = 99
    with pytest.raises(ProviderError, match="inventory manifest mismatch"):
        result.assert_manifest(modified)


def test_exact_budget_boundary_and_client_use_identical_cost_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret")
    spec = _spec(tmp_path, price=TokenPrice(2.0, 3.0))
    exact = spec.conservative_cost_usd
    ledger = _ledger(tmp_path / "cost.yaml", api=exact, total=exact)
    result = preflight_openrouter_phase(
        phase="behavior_baseline_api",
        requests=(spec,),
        ledger=ledger,
    )
    assert result.manifest["additional_commitment_required_usd"] == exact
    assert result.manifest["ledger"]["projected_after_completion_usd"]["api"] == exact

    client = OpenRouterJSONClient(
        model_id=spec.model_id,
        price=spec.price,
        ledger=ledger,
        max_output_tokens=spec.max_output_tokens,
        paid_response_store=spec.paid_response_store,
        transport=lambda *_: HTTPResult(
            status=200,
            body={
                "id": "response",
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "cost": 0.0},
                "choices": [{"message": {"content": "{}"}}],
            },
        ),
    )
    assert (
        client.complete_json(
            request_id=spec.request_id,
            system_prompt=spec.system_prompt,
            user_content=spec.user_content,
            purpose=spec.purpose,
        )
        == "{}"
    )
    assert round(float(client.metadata["preflight_upper_bound_usd"]), 6) == exact


def test_fractional_micro_cost_is_ceiled_identically_at_preflight_and_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret")
    store = PaidResponseStore(tmp_path / "responses")
    spec = OpenRouterRequestSpec(
        route="primary",
        model_id="provider/model",
        price=TokenPrice(0.1, 0.1),
        request_id="fractional-micro",
        purpose="adjudication",
        user_content="payload",
        max_output_tokens=1,
        paid_response_store=store,
    )
    assert spec.conservative_cost_usd == 0.000008
    ledger = _ledger(tmp_path / "cost.yaml", api=spec.conservative_cost_usd, total=1.0)
    preflight = preflight_openrouter_phase(
        phase="positions_api",
        requests=(spec,),
        ledger=ledger,
    )
    client = OpenRouterJSONClient(
        model_id=spec.model_id,
        price=spec.price,
        ledger=ledger,
        max_output_tokens=spec.max_output_tokens,
        paid_response_store=store,
        dispatch_guard=OpenRouterDispatchGuard(preflight),
        dispatch_route=spec.route,
        transport=lambda *_args: HTTPResult(
            status=200,
            body={
                "choices": [{"message": {"content": "{}"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "cost": 0.0},
            },
        ),
    )

    assert (
        client.complete_json(
            request_id=spec.request_id,
            user_content=spec.user_content,
            purpose=spec.purpose,
        )
        == "{}"
    )
    assert client.metadata["preflight_upper_bound_usd"] == spec.conservative_cost_usd


def test_paid_receipt_explicitly_binds_completion_inventory(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    completion = preflight_openrouter_phase(
        phase="behavior_baseline_api",
        requests=(spec,),
        ledger=_ledger(tmp_path / "cost.yaml"),
    )
    receipt = PaidPhaseReceiptStore(tmp_path / "receipts").authorize(
        command_phase="behavior_baseline_api",
        approval_content_hash=stable_hash("approval"),
        approval_id_hash=stable_hash("approval-id"),
        bindings_hash=stable_hash("bindings"),
        plan_hash=stable_hash("plan"),
        api_completion_preflight=completion.manifest,
    )
    bound = receipt["api_completion_preflight"]
    assert bound["manifest_hash"] == completion.manifest_hash
    assert bound["full_inventory_hash"] == completion.manifest["full_inventory_hash"]
    assert bound["pending_inventory_hash"] == completion.manifest["pending_inventory_hash"]
    assert bound["conservative_pending_usd"] == spec.conservative_cost_usd
    assert bound["per_route"] == completion.manifest["per_route"]
    assert bound["ledger"] == completion.manifest["ledger"]


def test_paid_plan_resume_accepts_only_monotone_authenticated_cache_subset(
    tmp_path: Path,
) -> None:
    first_store = PaidResponseStore(tmp_path / "responses")
    first_spec = _spec(tmp_path, store=first_store)
    second_spec = OpenRouterRequestSpec(
        route="primary",
        model_id=first_spec.model_id,
        model_revision=None,
        price=first_spec.price,
        request_id="request-2",
        purpose=first_spec.purpose,
        system_prompt=first_spec.system_prompt,
        user_content="second payload",
        max_output_tokens=first_spec.max_output_tokens,
        paid_response_store=first_store,
    )
    ledger = _ledger(tmp_path / "cost.yaml")
    initial = preflight_openrouter_phase(
        phase="behavior_baseline_api",
        requests=(first_spec, second_spec),
        ledger=ledger,
    )
    path = tmp_path / "paid_plan.json"
    frozen = cli._freeze_or_reuse_api_paid_plan(
        path,
        cli._bind_api_completion_preflight(
            {"protocol_version": "test-plan-v1", "plan_hash": stable_hash("old")},
            initial,
        ),
        initial,
        label="test paid plan",
    )

    first_store.commit(
        key=first_spec.store_key,
        request_fingerprint=first_spec.request_fingerprint,
        logical_request_hash=stable_hash(first_spec.request_id),
        model_id=first_spec.model_id,
        purpose=first_spec.purpose,
        http_status=200,
        response_body={
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "cost": 0.0},
            "choices": [{"message": {"content": "{}"}}],
        },
    )
    resumed = preflight_openrouter_phase(
        phase="behavior_baseline_api",
        requests=(first_spec, second_spec),
        ledger=ledger,
    )
    reused = cli._freeze_or_reuse_api_paid_plan(
        path,
        cli._bind_api_completion_preflight(
            {"protocol_version": "test-plan-v1", "plan_hash": stable_hash("old")},
            resumed,
        ),
        resumed,
        label="test paid plan",
    )
    assert reused == frozen
    assert resumed.manifest["pending_request_count"] == 1
    attempt = cli._freeze_api_completion_attempt(
        tmp_path,
        paid_plan_hash=str(reused["plan_hash"]),
        preflight=resumed,
    )
    assert attempt["api_completion_preflight"]["manifest_hash"] == resumed.manifest_hash


def test_cached_checkpoint_without_matching_ledger_reservation_fails_closed(
    tmp_path: Path,
) -> None:
    store = PaidResponseStore(tmp_path / "responses")
    spec = _spec(tmp_path, store=store)
    store.commit(
        key=spec.store_key,
        request_fingerprint=spec.request_fingerprint,
        logical_request_hash=stable_hash(spec.request_id),
        model_id=spec.model_id,
        purpose=spec.purpose,
        http_status=200,
        response_body={
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "cost": 0.0},
            "choices": [{"message": {"content": "{}"}}],
        },
    )

    with pytest.raises(ProviderError, match="ledger reservation"):
        preflight_openrouter_phase(
            phase="positions_api",
            requests=(spec,),
            ledger=_ledger(tmp_path / "cost.yaml"),
        )


def test_dispatch_guard_blocks_unplanned_exact_request_before_transport(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret")
    spec = _spec(tmp_path)
    ledger = _ledger(tmp_path / "cost.yaml")
    preflight = preflight_openrouter_phase(phase="positions_api", requests=(spec,), ledger=ledger)
    called = False

    def transport(*_args):
        nonlocal called
        called = True
        raise AssertionError("unplanned transport must not run")

    client = OpenRouterJSONClient(
        model_id=spec.model_id,
        price=spec.price,
        ledger=ledger,
        max_output_tokens=spec.max_output_tokens,
        paid_response_store=spec.paid_response_store,
        dispatch_guard=OpenRouterDispatchGuard(preflight),
        dispatch_route=spec.route,
        transport=transport,
    )
    with pytest.raises(ProviderError, match="not present in the authorized phase inventory"):
        client.complete_json(
            request_id="unplanned-request",
            system_prompt=spec.system_prompt,
            user_content=spec.user_content,
            purpose=spec.purpose,
        )
    assert called is False


def test_guarded_cached_request_cannot_transport_after_checkpoint_disappears_at_zero_headroom(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret")
    store = PaidResponseStore(tmp_path / "responses")
    spec = _spec(tmp_path, store=store, price=TokenPrice(1_000_000.0, 1_000_000.0))
    cap = spec.conservative_cost_usd
    ledger = _ledger(tmp_path / "cost.yaml", api=cap, total=cap)
    ledger.reserve(
        spec.reservation_id,
        CostEntry(
            kind="api",
            amount_usd=cap,
            description=f"preflight OpenRouter {spec.purpose} {stable_hash(spec.request_id)}",
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
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "cost": cap},
            "choices": [{"message": {"content": "{}"}}],
        },
    )
    ledger.settle_reservation(
        spec.reservation_id,
        CostEntry(
            kind="api",
            amount_usd=cap,
            description=f"OpenRouter {spec.purpose} {stable_hash(spec.request_id)}",
        ),
    )
    preflight = preflight_openrouter_phase(phase="positions_api", requests=(spec,), ledger=ledger)
    checkpoint = store.directory / f"{spec.store_key.split(':', 1)[1]}.json"
    checkpoint.unlink()
    called = False

    def transport(*_args):
        nonlocal called
        called = True
        raise AssertionError("a request frozen as cached must never transport")

    client = OpenRouterJSONClient(
        model_id=spec.model_id,
        price=spec.price,
        ledger=ledger,
        max_output_tokens=spec.max_output_tokens,
        paid_response_store=store,
        dispatch_guard=OpenRouterDispatchGuard(preflight),
        dispatch_route=spec.route,
        transport=transport,
    )
    with pytest.raises(ProviderError, match="frozen as cached"):
        client.complete_json(
            request_id=spec.request_id,
            system_prompt=spec.system_prompt,
            user_content=spec.user_content,
            purpose=spec.purpose,
        )
    assert called is False


def test_dispatch_guard_rejects_paid_response_store_swap_before_transport(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret")
    original_store = PaidResponseStore(tmp_path / "responses-a")
    spec = _spec(tmp_path, store=original_store)
    ledger = _ledger(tmp_path / "cost.yaml")
    preflight = preflight_openrouter_phase(phase="positions_api", requests=(spec,), ledger=ledger)
    dispatch_guard = OpenRouterDispatchGuard(preflight)
    called = False

    def transport(*_args):
        nonlocal called
        called = True
        raise AssertionError("swapped store must fail before transport")

    archived_store = tmp_path / "responses-a-before-swap"
    original_store.directory.rename(archived_store)
    swapped_store = PaidResponseStore(original_store.directory)
    client = OpenRouterJSONClient(
        model_id=spec.model_id,
        price=spec.price,
        ledger=ledger,
        max_output_tokens=spec.max_output_tokens,
        paid_response_store=swapped_store,
        dispatch_guard=dispatch_guard,
        dispatch_route=spec.route,
        transport=transport,
    )
    with pytest.raises(ProviderError, match="authorized phase inventory"):
        client.complete_json(
            request_id=spec.request_id,
            system_prompt=spec.system_prompt,
            user_content=spec.user_content,
            purpose=spec.purpose,
        )
    assert called is False


def test_dispatch_rejects_inode_swap_between_guard_and_checkpoint_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret")
    store = PaidResponseStore(tmp_path / "responses")
    spec = _spec(tmp_path, store=store)
    ledger = _ledger(tmp_path / "cost.yaml")
    preflight = preflight_openrouter_phase(phase="positions_api", requests=(spec,), ledger=ledger)
    real_load = store.load
    load_calls = 0

    def swap_during_checkpoint_load(*, key: str, request_fingerprint: str):
        nonlocal load_calls
        load_calls += 1
        if load_calls == 1:
            store.directory.rename(tmp_path / "responses-before-swap")
            store.directory.mkdir()
        return real_load(key=key, request_fingerprint=request_fingerprint)

    monkeypatch.setattr(store, "load", swap_during_checkpoint_load)
    transports = 0

    def transport(*_args):
        nonlocal transports
        transports += 1
        raise AssertionError("an inode-swapped response store must fail before transport")

    client = OpenRouterJSONClient(
        model_id=spec.model_id,
        price=spec.price,
        ledger=ledger,
        max_output_tokens=spec.max_output_tokens,
        paid_response_store=store,
        dispatch_guard=OpenRouterDispatchGuard(preflight),
        dispatch_route=spec.route,
        transport=transport,
    )
    with pytest.raises(PaidResponseStoreError, match="anchored directory"):
        client.complete_json(
            request_id=spec.request_id,
            system_prompt=spec.system_prompt,
            user_content=spec.user_content,
            purpose=spec.purpose,
        )
    assert transports == 0


@pytest.mark.parametrize("control_name", [".claims", ".uncertain"])
def test_dispatch_rejects_control_directory_inode_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    control_name: str,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret")
    store = PaidResponseStore(tmp_path / "responses")
    spec = _spec(tmp_path, store=store)
    ledger = _ledger(tmp_path / "cost.yaml")
    preflight = preflight_openrouter_phase(phase="positions_api", requests=(spec,), ledger=ledger)
    dispatch_guard = OpenRouterDispatchGuard(preflight)
    control_path = store.directory / control_name
    control_path.rename(store.directory / f"{control_name}.before-swap")
    control_path.mkdir()
    transports = 0

    def transport(*_args):
        nonlocal transports
        transports += 1
        raise AssertionError("a control-directory swap must fail before transport")

    client = OpenRouterJSONClient(
        model_id=spec.model_id,
        price=spec.price,
        ledger=ledger,
        max_output_tokens=spec.max_output_tokens,
        paid_response_store=store,
        dispatch_guard=dispatch_guard,
        dispatch_route=spec.route,
        transport=transport,
    )
    with pytest.raises(PaidResponseStoreError, match="anchored directory"):
        client.complete_json(
            request_id=spec.request_id,
            system_prompt=spec.system_prompt,
            user_content=spec.user_content,
            purpose=spec.purpose,
        )
    assert transports == 0


def test_request_frozen_pending_may_replay_checkpoint_created_before_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret")
    store = PaidResponseStore(tmp_path / "responses")
    spec = _spec(tmp_path, store=store)
    ledger = _ledger(tmp_path / "cost.yaml")
    preflight = preflight_openrouter_phase(phase="positions_api", requests=(spec,), ledger=ledger)
    store.commit(
        key=spec.store_key,
        request_fingerprint=spec.request_fingerprint,
        logical_request_hash=stable_hash(spec.request_id),
        model_id=spec.model_id,
        purpose=spec.purpose,
        http_status=200,
        response_body={
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "cost": 0.0},
            "choices": [{"message": {"content": "{}"}}],
        },
    )
    client = OpenRouterJSONClient(
        model_id=spec.model_id,
        price=spec.price,
        ledger=ledger,
        max_output_tokens=spec.max_output_tokens,
        paid_response_store=store,
        dispatch_guard=OpenRouterDispatchGuard(preflight),
        dispatch_route=spec.route,
        transport=lambda *_args: (_ for _ in ()).throw(AssertionError("must replay")),
    )
    assert (
        client.complete_json(
            request_id=spec.request_id,
            system_prompt=spec.system_prompt,
            user_content=spec.user_content,
            purpose=spec.purpose,
        )
        == "{}"
    )


def test_paid_plan_rejects_swapped_pending_identity_even_when_counts_match(
    tmp_path: Path,
) -> None:
    store = PaidResponseStore(tmp_path / "responses")
    first = _spec(tmp_path, store=store)
    second = OpenRouterRequestSpec(
        route=first.route,
        model_id=first.model_id,
        model_revision=None,
        price=first.price,
        request_id="request-2",
        purpose=first.purpose,
        system_prompt=first.system_prompt,
        user_content="second payload",
        max_output_tokens=first.max_output_tokens,
        paid_response_store=store,
    )
    ledger = _ledger(tmp_path / "cost.yaml")
    initial = preflight_openrouter_phase(
        phase="behavior_baseline_api", requests=(first, second), ledger=ledger
    )
    path = tmp_path / "paid_plan.json"
    cli._freeze_or_reuse_api_paid_plan(
        path,
        cli._bind_api_completion_preflight(
            {"protocol_version": "test-plan-v1", "plan_hash": stable_hash("old")},
            initial,
        ),
        initial,
        label="test paid plan",
    )
    # Authenticate only request 1, making {request 2} the exact live pending set.
    store.commit(
        key=first.store_key,
        request_fingerprint=first.request_fingerprint,
        logical_request_hash=stable_hash(first.request_id),
        model_id=first.model_id,
        purpose=first.purpose,
        http_status=200,
        response_body={
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "cost": 0.0},
            "choices": [{"message": {"content": "{}"}}],
        },
    )
    one_pending = preflight_openrouter_phase(
        phase="behavior_baseline_api", requests=(first, second), ledger=ledger
    )
    cli._freeze_api_completion_attempt(
        tmp_path,
        paid_plan_hash=stable_hash("paid-plan"),
        preflight=one_pending,
    )
    # Swap to {request 1} pending by moving the authenticated checkpoint.
    first_path = store.directory / f"{first.store_key.split(':', 1)[1]}.json"
    second_path = store.directory / f"{second.store_key.split(':', 1)[1]}.json"
    first_path.replace(second_path)
    changed = json.loads(second_path.read_text(encoding="utf-8"))
    changed["store_key"] = second.store_key
    changed["request_fingerprint"] = second.request_fingerprint
    changed["logical_request_hash"] = stable_hash(second.request_id)
    changed["record_hash"] = stable_hash(
        {key: value for key, value in changed.items() if key != "record_hash"}
    )
    write_json(second_path, changed)
    swapped = preflight_openrouter_phase(
        phase="behavior_baseline_api", requests=(first, second), ledger=ledger
    )
    with pytest.raises(cli.CLIError, match="not a monotone subset"):
        cli._freeze_api_completion_attempt(
            tmp_path,
            paid_plan_hash=stable_hash("paid-plan"),
            preflight=swapped,
        )
