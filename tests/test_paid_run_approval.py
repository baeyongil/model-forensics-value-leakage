from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from model_forensics.approval import (
    APPROVAL_FILENAME,
    APPROVAL_SCHEMA_VERSION,
    PHASE_CONTRACT_VERSION,
    ApiQuoteBinding,
    ApprovalBindings,
    GpuBinding,
    GpuPhaseRuntimeAllocation,
    GpuQuote,
    PaidRunApproval,
    PaidRunApprovalError,
    RouteBinding,
    SpendingCaps,
    UserApproval,
    approval_content_hash,
    load_paid_run_approval,
    validate_paid_run_approval,
)
from model_forensics.io import stable_hash, write_json

NOW = datetime(2026, 8, 29, 18, tzinfo=UTC)
SHA_A = stable_hash({"fixture": "frozen-config"})
SHA_B = stable_hash({"fixture": "frozen-preregistration"})
SHA_C = stable_hash({"fixture": "mutated-config"})
SHA_D = stable_hash({"fixture": "mutated-preregistration"})
SHA_GPU_LOCK = stable_hash({"fixture": "gpu-lock"})
SHA_OTHER_GPU_LOCK = stable_hash({"fixture": "other-gpu-lock"})
RAW_C = stable_hash({"fixture": "container"}).removeprefix("sha256:")
RAW_D = stable_hash({"fixture": "wheel"}).removeprefix("sha256:")
RAW_E = stable_hash({"fixture": "other-container"}).removeprefix("sha256:")
RAW_F = stable_hash({"fixture": "other-wheel"}).removeprefix("sha256:")
GPU_QUOTE_HASH = stable_hash({"fixture": "gpu-quote"})
API_QUOTE_HASH = stable_hash({"fixture": "api-quote"})


def _phase_allocations() -> tuple[GpuPhaseRuntimeAllocation, ...]:
    return (
        GpuPhaseRuntimeAllocation(command_phase="behavior_baseline_gpu", maximum_runtime_hours=1.5),
        GpuPhaseRuntimeAllocation(
            command_phase="behavior_treatment_gpu", maximum_runtime_hours=2.0
        ),
        GpuPhaseRuntimeAllocation(command_phase="resample_gpu", maximum_runtime_hours=3.0),
        GpuPhaseRuntimeAllocation(command_phase="lens_gpu", maximum_runtime_hours=2.0),
    )


def _bindings() -> ApprovalBindings:
    return ApprovalBindings(
        phase_contract_version=PHASE_CONTRACT_VERSION,
        config_hash=SHA_A,
        preregistration_hash=SHA_B,
        gpu_lock_hash=SHA_GPU_LOCK,
        gpu=GpuBinding(
            family="H100_80GB",
            provider_gpu_id="NVIDIA H100 80GB HBM3",
            cloud_type="SECURE",
            allowed_cuda_versions=("12.8",),
            data_center_ids=("US-IL-1",),
            count=8,
            container_disk_gb=50,
            volume_disk_gb=650,
            quote=GpuQuote(
                provider="runpod",
                quote_id="runpod-secure-h100-20260829",
                usd_per_gpu_hour=3.0,
                running_storage_usd_per_hour=700 * 0.10 / 720,
                quoted_at=NOW - timedelta(hours=1),
                source_url="https://www.runpod.io/pricing",
                content_hash=GPU_QUOTE_HASH,
            ),
            phase_runtime_allocations=_phase_allocations(),
            container_image_digest=f"vllm/vllm-openai@sha256:{RAW_C}",
            vllm_wheel_sha256=RAW_D,
        ),
        api_quote=ApiQuoteBinding(
            provider="openrouter",
            source_url="https://openrouter.ai/models",
            checked_at=NOW - timedelta(minutes=45),
            content_hash=API_QUOTE_HASH,
        ),
        caps_usd=SpendingCaps(gpu=220.0, api=100.0, total=325.0),
        routes=(
            RouteBinding(
                role="primary_final_and_trajectory",
                provider="openrouter",
                model="anthropic/claude-opus-5",
                input_usd_per_million_tokens=5.0,
                output_usd_per_million_tokens=25.0,
            ),
            RouteBinding(
                role="independent_final",
                provider="openrouter",
                model="google/gemini-3.1-pro-preview",
                input_usd_per_million_tokens=2.0,
                output_usd_per_million_tokens=12.0,
            ),
            RouteBinding(
                role="classifier_anthropic",
                provider="openrouter",
                model="anthropic/claude-opus-5",
                input_usd_per_million_tokens=5.0,
                output_usd_per_million_tokens=25.0,
            ),
            RouteBinding(
                role="classifier_google",
                provider="openrouter",
                model="google/gemini-3.1-pro-preview",
                input_usd_per_million_tokens=2.0,
                output_usd_per_million_tokens=12.0,
            ),
        ),
    )


def _approval(bindings: ApprovalBindings | None = None) -> PaidRunApproval:
    document = PaidRunApproval(
        schema_version=APPROVAL_SCHEMA_VERSION,
        bindings=bindings or _bindings(),
        allowed_command_phases=("behavior_baseline_gpu", "behavior_baseline_api"),
        user_approval=UserApproval(
            approval_id="approval-yib-20260829-01",
            approved_at=NOW - timedelta(minutes=30),
        ),
        content_hash=SHA_A,
    )
    return document.model_copy(
        update={"content_hash": approval_content_hash(document)},
    )


def _write_raw(tmp_path, raw: dict[str, object]):
    path = tmp_path / APPROVAL_FILENAME
    raw["content_hash"] = approval_content_hash(raw)
    write_json(path, raw)
    return path


def _replace_path(raw: dict[str, object], path: tuple[str | int, ...], value: object) -> None:
    target: object = raw
    for component in path[:-1]:
        target = target[component]  # type: ignore[index]
    target[path[-1]] = value  # type: ignore[index]


def test_load_and_validate_exact_content_addressed_approval(tmp_path) -> None:
    path = tmp_path / APPROVAL_FILENAME
    approval = _approval()
    write_json(path, approval.model_dump(mode="json"))

    loaded = load_paid_run_approval(path)
    validated = validate_paid_run_approval(
        loaded,
        expected=_bindings(),
        command_phase="behavior_baseline_gpu",
        now=NOW,
    )

    assert validated == approval


def test_validate_rechecks_content_hash_even_for_in_memory_document() -> None:
    approval = _approval().model_copy(update={"content_hash": SHA_A})
    with pytest.raises(PaidRunApprovalError, match="content hash"):
        validate_paid_run_approval(
            approval,
            expected=_bindings(),
            command_phase="behavior_baseline_gpu",
            now=NOW,
        )


def test_strict_schema_rejects_coerced_numeric_values(tmp_path) -> None:
    raw = _approval().model_dump(mode="json")
    _replace_path(raw, ("bindings", "gpu", "count"), "8")
    path = _write_raw(tmp_path, raw)

    with pytest.raises(PaidRunApprovalError, match="schema"):
        load_paid_run_approval(path)


def test_duplicate_json_keys_are_rejected(tmp_path) -> None:
    raw = _approval().model_dump(mode="json")
    encoded = json.dumps(raw)
    encoded = encoded.replace(
        f'"schema_version": {APPROVAL_SCHEMA_VERSION},',
        f'"schema_version": {APPROVAL_SCHEMA_VERSION}, '
        f'"schema_version": {APPROVAL_SCHEMA_VERSION},',
        1,
    )
    path = tmp_path / APPROVAL_FILENAME
    path.write_text(encoded, encoding="utf-8")

    with pytest.raises(PaidRunApprovalError, match="duplicate"):
        load_paid_run_approval(path)


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("bindings", "config_hash"), SHA_C),
        (("bindings", "preregistration_hash"), SHA_D),
        (("bindings", "gpu_lock_hash"), SHA_OTHER_GPU_LOCK),
        (("bindings", "gpu", "family"), "A100_80GB"),
        (("bindings", "gpu", "provider_gpu_id"), "NVIDIA H100 SXM 80GB"),
        (("bindings", "gpu", "count"), 4),
        (("bindings", "gpu", "quote", "provider"), "another-cloud"),
        (("bindings", "gpu", "quote", "quote_id"), "another-quote-20260829"),
        (("bindings", "gpu", "quote", "usd_per_gpu_hour"), 2.9),
        (("bindings", "gpu", "quote", "running_storage_usd_per_hour"), 0.09),
        (
            ("bindings", "gpu", "quote", "quoted_at"),
            (NOW - timedelta(hours=2)).isoformat(),
        ),
        (("bindings", "gpu", "quote", "source_url"), "https://cloud.google.com/gpu"),
        (("bindings", "gpu", "quote", "content_hash"), SHA_C),
        (
            ("bindings", "gpu", "phase_runtime_allocations", 1, "maximum_runtime_hours"),
            1.9,
        ),
        (
            ("bindings", "gpu", "container_image_digest"),
            f"vllm/vllm-openai@sha256:{RAW_E}",
        ),
        (("bindings", "gpu", "vllm_wheel_sha256"), RAW_F),
        (("bindings", "caps_usd", "gpu"), 219.0),
        (("bindings", "caps_usd", "api"), 99.0),
        (("bindings", "caps_usd", "total"), 326.0),
        (("bindings", "api_quote", "provider"), "another-gateway"),
        (("bindings", "api_quote", "source_url"), "https://platform.openai.com/pricing"),
        (
            ("bindings", "api_quote", "checked_at"),
            (NOW - timedelta(hours=2)).isoformat(),
        ),
        (("bindings", "api_quote", "content_hash"), SHA_D),
        (("bindings", "routes", 0, "model"), "anthropic/claude-opus-4.1"),
        (("bindings", "routes", 1, "input_usd_per_million_tokens"), 2.1),
        (("bindings", "routes", 3, "output_usd_per_million_tokens"), 12.1),
    ],
)
def test_every_paid_binding_must_exactly_match_frozen_expectations(
    tmp_path,
    path: tuple[str | int, ...],
    replacement: object,
) -> None:
    raw = _approval().model_dump(mode="json")
    _replace_path(raw, path, replacement)
    approval = load_paid_run_approval(_write_raw(tmp_path, raw))

    with pytest.raises(PaidRunApprovalError, match="bindings"):
        validate_paid_run_approval(
            approval,
            expected=_bindings(),
            command_phase="behavior_baseline_gpu",
            now=NOW,
        )


def test_route_removal_and_extra_route_are_binding_mismatches(tmp_path) -> None:
    for operation in ("remove", "add"):
        raw = _approval().model_dump(mode="json")
        routes = raw["bindings"]["routes"]  # type: ignore[index]
        if operation == "remove":
            routes.pop()  # type: ignore[union-attr]
        else:
            routes.append(  # type: ignore[union-attr]
                {
                    "role": "unapproved_tiebreaker",
                    "provider": "openrouter",
                    "model": "anthropic/claude-opus-5",
                    "input_usd_per_million_tokens": 5.0,
                    "output_usd_per_million_tokens": 25.0,
                }
            )
        approval = load_paid_run_approval(_write_raw(tmp_path, raw))
        with pytest.raises(PaidRunApprovalError, match="bindings"):
            validate_paid_run_approval(
                approval,
                expected=_bindings(),
                command_phase="behavior_baseline_gpu",
                now=NOW,
            )


def test_unsupported_phase_contract_version_is_rejected(tmp_path) -> None:
    raw = _approval().model_dump(mode="json")
    _replace_path(raw, ("bindings", "phase_contract_version"), "gpu-api-phase-split-v3")
    with pytest.raises(PaidRunApprovalError, match="schema"):
        load_paid_run_approval(_write_raw(tmp_path, raw))


def test_obsolete_approval_schema_version_is_rejected(tmp_path) -> None:
    raw = _approval().model_dump(mode="json")
    raw["schema_version"] = 1
    with pytest.raises(PaidRunApprovalError, match="schema"):
        load_paid_run_approval(_write_raw(tmp_path, raw))


def test_unlisted_command_phase_is_rejected() -> None:
    with pytest.raises(PaidRunApprovalError, match="not approved"):
        validate_paid_run_approval(
            _approval(),
            expected=_bindings(),
            command_phase="behavior_treatment_gpu",
            now=NOW,
        )


def test_noncanonical_command_phase_cannot_enter_approval_document(tmp_path) -> None:
    raw = _approval().model_dump(mode="json")
    raw["allowed_command_phases"] = ["custom_paid_phase"]
    with pytest.raises(PaidRunApprovalError, match="canonical"):
        load_paid_run_approval(_write_raw(tmp_path, raw))


def test_quote_older_than_six_hours_is_rejected(tmp_path) -> None:
    raw = _approval().model_dump(mode="json")
    _replace_path(
        raw,
        ("bindings", "gpu", "quote", "quoted_at"),
        (NOW - timedelta(hours=6, seconds=1)).isoformat(),
    )
    approval = load_paid_run_approval(_write_raw(tmp_path, raw))
    with pytest.raises(PaidRunApprovalError, match="older than six hours"):
        validate_paid_run_approval(
            approval,
            expected=approval.bindings,
            command_phase="behavior_baseline_gpu",
            now=NOW,
        )


def test_stale_gpu_quote_does_not_block_already_approved_api_only_phase(tmp_path) -> None:
    raw = _approval().model_dump(mode="json")
    _replace_path(
        raw,
        ("bindings", "gpu", "quote", "quoted_at"),
        (NOW - timedelta(hours=8)).isoformat(),
    )
    raw["allowed_command_phases"] = ["behavior_baseline_api"]
    approval = load_paid_run_approval(_write_raw(tmp_path, raw))
    assert (
        validate_paid_run_approval(
            approval,
            expected=approval.bindings,
            command_phase="behavior_baseline_api",
            now=NOW,
        )
        == approval
    )


def test_stale_api_quote_is_rejected_only_for_api_phases(tmp_path) -> None:
    raw = _approval().model_dump(mode="json")
    _replace_path(
        raw,
        ("bindings", "api_quote", "checked_at"),
        (NOW - timedelta(hours=6, seconds=1)).isoformat(),
    )
    raw["allowed_command_phases"] = ["behavior_baseline_gpu", "behavior_baseline_api"]
    approval = load_paid_run_approval(_write_raw(tmp_path, raw))

    with pytest.raises(PaidRunApprovalError, match="API quote is older than six hours"):
        validate_paid_run_approval(
            approval,
            expected=approval.bindings,
            command_phase="behavior_baseline_api",
            now=NOW,
        )
    assert (
        validate_paid_run_approval(
            approval,
            expected=approval.bindings,
            command_phase="behavior_baseline_gpu",
            now=NOW,
        )
        == approval
    )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda rows: rows.pop(),
        lambda rows: rows.append(dict(rows[0])),
        lambda rows: rows.__setitem__(
            0,
            {
                "command_phase": "custom_gpu",
                "maximum_runtime_hours": 1.5,
            },
        ),
    ],
)
def test_gpu_runtime_allocations_require_each_canonical_phase_exactly_once(
    tmp_path, mutate
) -> None:  # type: ignore[no-untyped-def]
    raw = _approval().model_dump(mode="json")
    allocations = raw["bindings"]["gpu"]["phase_runtime_allocations"]  # type: ignore[index]
    mutate(allocations)

    with pytest.raises(PaidRunApprovalError, match="schema"):
        load_paid_run_approval(_write_raw(tmp_path, raw))


def test_sum_of_gpu_phase_allocations_must_fit_gpu_cap(tmp_path) -> None:
    raw = _approval().model_dump(mode="json")
    allocations = raw["bindings"]["gpu"]["phase_runtime_allocations"]  # type: ignore[index]
    allocations[2]["maximum_runtime_hours"] = 4.0  # type: ignore[index]

    with pytest.raises(PaidRunApprovalError, match="GPU cap"):
        load_paid_run_approval(_write_raw(tmp_path, raw))


def test_user_approval_must_follow_quote_and_not_be_in_the_future(tmp_path) -> None:
    raw = _approval().model_dump(mode="json")
    _replace_path(
        raw,
        ("user_approval", "approved_at"),
        (NOW - timedelta(hours=2)).isoformat(),
    )
    approval = load_paid_run_approval(_write_raw(tmp_path, raw))
    with pytest.raises(PaidRunApprovalError, match="predates"):
        validate_paid_run_approval(
            approval,
            expected=approval.bindings,
            command_phase="behavior_baseline_gpu",
            now=NOW,
        )

    raw = _approval().model_dump(mode="json")
    _replace_path(
        raw,
        ("user_approval", "approved_at"),
        (NOW + timedelta(minutes=6)).isoformat(),
    )
    approval = load_paid_run_approval(_write_raw(tmp_path, raw))
    with pytest.raises(PaidRunApprovalError, match="future"):
        validate_paid_run_approval(
            approval,
            expected=approval.bindings,
            command_phase="behavior_baseline_gpu",
            now=NOW,
        )


def test_user_approval_must_follow_api_quote_check(tmp_path) -> None:
    raw = _approval().model_dump(mode="json")
    _replace_path(
        raw,
        ("bindings", "api_quote", "checked_at"),
        (NOW - timedelta(minutes=20)).isoformat(),
    )
    approval = load_paid_run_approval(_write_raw(tmp_path, raw))

    with pytest.raises(PaidRunApprovalError, match="predates the API quote"):
        validate_paid_run_approval(
            approval,
            expected=approval.bindings,
            command_phase="behavior_baseline_api",
            now=NOW,
        )


def test_approval_chronology_is_checked_only_against_the_phase_relevant_quote(
    tmp_path,
) -> None:
    api_raw = _approval().model_dump(mode="json")
    _replace_path(
        api_raw,
        ("bindings", "gpu", "quote", "quoted_at"),
        (NOW - timedelta(minutes=20)).isoformat(),
    )
    api_raw["allowed_command_phases"] = ["behavior_baseline_api"]
    api_approval = load_paid_run_approval(_write_raw(tmp_path, api_raw))
    assert (
        validate_paid_run_approval(
            api_approval,
            expected=api_approval.bindings,
            command_phase="behavior_baseline_api",
            now=NOW,
        )
        == api_approval
    )

    gpu_raw = _approval().model_dump(mode="json")
    _replace_path(
        gpu_raw,
        ("bindings", "api_quote", "checked_at"),
        (NOW - timedelta(minutes=20)).isoformat(),
    )
    gpu_raw["allowed_command_phases"] = ["behavior_baseline_gpu"]
    gpu_approval = load_paid_run_approval(_write_raw(tmp_path, gpu_raw))
    assert (
        validate_paid_run_approval(
            gpu_approval,
            expected=gpu_approval.bindings,
            command_phase="behavior_baseline_gpu",
            now=NOW,
        )
        == gpu_approval
    )


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("user_approval", "approval_id"), "approval-placeholder"),
        (("user_approval", "approval_id"), "sk-abcdefghijklmnop"),
        (("bindings", "phase_contract_version"), "pending"),
        (("bindings", "gpu", "family"), "<GPU_FAMILY>"),
        (("bindings", "gpu", "family"), "YOUR_GPU_FAMILY"),
        (("bindings", "gpu", "quote", "quote_id"), "TODO"),
        (("allowed_command_phases", 0), "replace_me"),
    ],
)
def test_placeholders_and_secret_like_approval_ids_are_rejected(
    tmp_path,
    path: tuple[str | int, ...],
    replacement: object,
) -> None:
    raw = _approval().model_dump(mode="json")
    _replace_path(raw, path, replacement)
    with pytest.raises(PaidRunApprovalError, match="schema"):
        load_paid_run_approval(_write_raw(tmp_path, raw))


@pytest.mark.parametrize(
    ("path", "extra"),
    [
        ((), {"surprise": True}),
        (("bindings",), {"surprise": True}),
        (("bindings", "gpu", "quote"), {"surprise": True}),
        (("bindings", "routes", 0), {"surprise": True}),
        (("user_approval",), {"surprise": True}),
    ],
)
def test_unknown_fields_are_rejected_at_every_schema_level(
    tmp_path,
    path: tuple[str | int, ...],
    extra: dict[str, object],
) -> None:
    raw = _approval().model_dump(mode="json")
    target: object = raw
    for component in path:
        target = target[component]  # type: ignore[index]
    target.update(extra)  # type: ignore[union-attr]
    with pytest.raises(PaidRunApprovalError, match="schema"):
        load_paid_run_approval(_write_raw(tmp_path, raw))


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("bindings", "config_hash"), "sha256:" + "0" * 64),
        (("bindings", "preregistration_hash"), "sha256:" + "f" * 64),
        (("bindings", "gpu_lock_hash"), "sha256:" + "e" * 64),
        (("bindings", "gpu", "vllm_wheel_sha256"), "a" * 64),
        (
            ("bindings", "gpu", "container_image_digest"),
            "vllm/vllm-openai@sha256:" + "b" * 64,
        ),
    ],
)
def test_degenerate_placeholder_hashes_are_rejected(
    tmp_path,
    path: tuple[str | int, ...],
    replacement: object,
) -> None:
    raw = _approval().model_dump(mode="json")
    _replace_path(raw, path, replacement)
    with pytest.raises(PaidRunApprovalError, match="schema"):
        load_paid_run_approval(_write_raw(tmp_path, raw))


def test_missing_user_approval_object_is_rejected(tmp_path) -> None:
    raw = _approval().model_dump(mode="json")
    del raw["user_approval"]
    with pytest.raises(PaidRunApprovalError, match="schema"):
        load_paid_run_approval(_write_raw(tmp_path, raw))


def test_tampering_and_missing_approval_file_are_rejected(tmp_path) -> None:
    missing = tmp_path / APPROVAL_FILENAME
    with pytest.raises(PaidRunApprovalError, match="missing"):
        load_paid_run_approval(missing)

    raw = _approval().model_dump(mode="json")
    raw["content_hash"] = SHA_B
    write_json(missing, raw)
    with pytest.raises(PaidRunApprovalError, match="content hash"):
        load_paid_run_approval(missing)


def test_approval_must_use_fixed_filename(tmp_path) -> None:
    path = tmp_path / "approval.json"
    write_json(path, _approval().model_dump(mode="json"))
    with pytest.raises(PaidRunApprovalError, match=APPROVAL_FILENAME):
        load_paid_run_approval(path)
