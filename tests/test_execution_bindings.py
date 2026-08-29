from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml

from model_forensics.config import load_preregistration, load_run_config
from model_forensics.execution_bindings import (
    API_ROUTE_QUOTE_LOCK_FILENAME,
    ApiRouteQuoteLockError,
    GpuQuoteLockError,
    api_route_quote_lock_content_hash,
    build_approval_bindings,
    gpu_quote_lock_content_hash,
    load_api_route_quote_lock,
    load_gpu_quote_lock,
)
from model_forensics.io import read_json, write_json

ROOT = Path(__file__).resolve().parents[1]


def _api_quote() -> dict:
    raw = {
        "schema_version": 1,
        "provider": "openrouter",
        "source_url": "https://openrouter.ai/models",
        "checked_at": datetime(2026, 8, 29, 18, tzinfo=UTC).isoformat(),
        "routes": [
            {
                "role": "primary_final_and_trajectory",
                "model": "anthropic/claude-opus-5",
                "input_usd_per_million_tokens": 5.0,
                "output_usd_per_million_tokens": 25.0,
            },
            {
                "role": "independent_final",
                "model": "google/gemini-3.1-pro-preview",
                "input_usd_per_million_tokens": 2.0,
                "output_usd_per_million_tokens": 12.0,
            },
            {
                "role": "classifier_anthropic",
                "model": "anthropic/claude-opus-5",
                "input_usd_per_million_tokens": 5.0,
                "output_usd_per_million_tokens": 25.0,
            },
            {
                "role": "classifier_google",
                "model": "google/gemini-3.1-pro-preview",
                "input_usd_per_million_tokens": 2.0,
                "output_usd_per_million_tokens": 12.0,
            },
        ],
    }
    raw["content_hash"] = api_route_quote_lock_content_hash(raw)
    return raw


def test_loads_exact_content_addressed_api_quote_lock(tmp_path: Path) -> None:
    path = tmp_path / API_ROUTE_QUOTE_LOCK_FILENAME
    write_json(path, _api_quote())

    quote = load_api_route_quote_lock(path)

    assert quote.provider == "openrouter"
    assert tuple(route.role for route in quote.routes) == (
        "primary_final_and_trajectory",
        "independent_final",
        "classifier_anthropic",
        "classifier_google",
    )
    assert quote.content_hash == _api_quote()["content_hash"]


def test_api_quote_lock_tampering_extra_fields_and_duplicate_keys_fail_closed(
    tmp_path: Path,
) -> None:
    path = tmp_path / API_ROUTE_QUOTE_LOCK_FILENAME
    write_json(path, _api_quote())
    tampered = read_json(path)
    tampered["routes"][0]["output_usd_per_million_tokens"] = 24.0
    write_json(path, tampered)
    with pytest.raises(ApiRouteQuoteLockError, match="content hash"):
        load_api_route_quote_lock(path)

    extra = _api_quote()
    extra["unfrozen_route"] = "forbidden"
    extra["content_hash"] = api_route_quote_lock_content_hash(extra)
    write_json(path, extra)
    with pytest.raises(ApiRouteQuoteLockError, match="schema"):
        load_api_route_quote_lock(path)

    encoded = '{"schema_version":1,"schema_version":1}'
    path.write_text(encoded, encoding="utf-8")
    with pytest.raises(ApiRouteQuoteLockError, match="duplicate"):
        load_api_route_quote_lock(path)


@pytest.mark.parametrize("operation", ["missing", "duplicate", "reordered", "unknown"])
def test_api_quote_lock_requires_exactly_four_canonical_routes(
    tmp_path: Path, operation: str
) -> None:
    raw = _api_quote()
    routes = raw["routes"]
    if operation == "missing":
        routes.pop()
    elif operation == "duplicate":
        routes[-1] = dict(routes[0])
    elif operation == "reordered":
        routes[0], routes[1] = routes[1], routes[0]
    else:
        routes[-1]["role"] = "unapproved_tiebreaker"
    raw["content_hash"] = api_route_quote_lock_content_hash(raw)
    path = tmp_path / API_ROUTE_QUOTE_LOCK_FILENAME
    write_json(path, raw)

    with pytest.raises(ApiRouteQuoteLockError, match="schema"):
        load_api_route_quote_lock(path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_url", "http://openrouter.ai/models"),
        ("source_url", "https://user:password@openrouter.ai/models"),
        ("checked_at", "2026-08-29T18:00:00"),
        ("provider", "another-gateway"),
    ],
)
def test_api_quote_lock_rejects_untrusted_provenance(
    tmp_path: Path, field: str, value: object
) -> None:
    raw = _api_quote()
    raw[field] = value
    raw["content_hash"] = api_route_quote_lock_content_hash(raw)
    path = tmp_path / API_ROUTE_QUOTE_LOCK_FILENAME
    write_json(path, raw)

    with pytest.raises(ApiRouteQuoteLockError, match="schema"):
        load_api_route_quote_lock(path)


def test_api_quote_lock_requires_fixed_filename(tmp_path: Path) -> None:
    path = tmp_path / "api-prices.json"
    write_json(path, _api_quote())
    with pytest.raises(ApiRouteQuoteLockError, match=API_ROUTE_QUOTE_LOCK_FILENAME):
        load_api_route_quote_lock(path)


def _quote() -> dict:
    raw = {
        "schema_version": 1,
        "provider": "runpod",
        "quote_id": "runpod-secure-h100-20260829-01",
        "gpu_family": "H100_80GB",
        "gpu_count": 8,
        "usd_per_gpu_hour": 3.0,
        "quoted_at": datetime(2026, 8, 29, 18, tzinfo=UTC).isoformat(),
        "phase_runtime_allocations": [
            {
                "command_phase": "behavior_baseline_gpu",
                "maximum_runtime_hours": 1.5,
            },
            {
                "command_phase": "behavior_treatment_gpu",
                "maximum_runtime_hours": 2.0,
            },
            {"command_phase": "resample_gpu", "maximum_runtime_hours": 3.0},
            {"command_phase": "lens_gpu", "maximum_runtime_hours": 2.0},
        ],
        "source_url": "https://www.runpod.io/pricing",
    }
    raw["content_hash"] = gpu_quote_lock_content_hash(raw)
    return raw


def _inputs(tmp_path: Path):  # type: ignore[no-untyped-def]
    config = load_run_config(ROOT / "config/run_122b.yaml")
    preregistration = load_preregistration(config)
    gpu_lock = yaml.safe_load((ROOT / "config/gpu_lock.yaml").read_text(encoding="utf-8"))
    path = tmp_path / "gpu_quote_lock.json"
    write_json(path, _quote())
    api_path = tmp_path / API_ROUTE_QUOTE_LOCK_FILENAME
    write_json(api_path, _api_quote())
    return (
        config,
        preregistration,
        gpu_lock,
        load_gpu_quote_lock(path),
        load_api_route_quote_lock(api_path),
    )


def test_bindings_are_built_independently_from_frozen_sources(tmp_path: Path) -> None:
    config, preregistration, gpu_lock, quote, api_quote = _inputs(tmp_path)
    bindings = build_approval_bindings(
        config=config,
        preregistration=preregistration,
        gpu_lock=gpu_lock,
        quote_lock=quote,
        api_quote_lock=api_quote,
    )

    assert bindings.gpu.family == "H100_80GB"
    assert bindings.gpu.count == 8
    assert bindings.gpu.container_image_digest.startswith("vllm/vllm-openai@sha256:")
    assert bindings.gpu.vllm_wheel_sha256 == gpu_lock["source_repositories"]["vllm"]["wheel_sha256"]
    assert bindings.gpu.quote.source_url == "https://www.runpod.io/pricing"
    assert bindings.gpu.quote.content_hash == quote.content_hash
    assert [
        (allocation.command_phase, allocation.maximum_runtime_hours)
        for allocation in bindings.gpu.phase_runtime_allocations
    ] == [
        ("behavior_baseline_gpu", 1.5),
        ("behavior_treatment_gpu", 2.0),
        ("resample_gpu", 3.0),
        ("lens_gpu", 2.0),
    ]
    assert bindings.api_quote.model_dump(mode="json") == {
        "provider": "openrouter",
        "source_url": "https://openrouter.ai/models",
        "checked_at": "2026-08-29T18:00:00Z",
        "content_hash": api_quote.content_hash,
    }
    assert bindings.caps_usd.model_dump() == {"gpu": 220.0, "api": 100.0, "total": 325.0}
    assert [(route.role, route.model) for route in bindings.routes] == [
        ("primary_final_and_trajectory", "anthropic/claude-opus-5"),
        ("independent_final", "google/gemini-3.1-pro-preview"),
        ("classifier_anthropic", "anthropic/claude-opus-5"),
        ("classifier_google", "google/gemini-3.1-pro-preview"),
    ]


def test_bindings_change_when_preregistered_route_price_changes(tmp_path: Path) -> None:
    config, preregistration, gpu_lock, quote, api_quote = _inputs(tmp_path)
    first = build_approval_bindings(
        config=config,
        preregistration=preregistration,
        gpu_lock=gpu_lock,
        quote_lock=quote,
        api_quote_lock=api_quote,
    )
    preregistration["external_judging"]["semantic_classification_routes"][1][
        "output_usd_per_million_tokens"
    ] = 12.5
    with pytest.raises(ValueError, match="API route quote"):
        build_approval_bindings(
            config=config,
            preregistration=preregistration,
            gpu_lock=gpu_lock,
            quote_lock=quote,
            api_quote_lock=api_quote,
        )
    assert first.preregistration_hash


def test_bindings_reject_live_api_quote_that_disagrees_with_preregistration(
    tmp_path: Path,
) -> None:
    config, preregistration, gpu_lock, quote, _ = _inputs(tmp_path)
    raw = _api_quote()
    raw["routes"][1]["input_usd_per_million_tokens"] = 2.1
    raw["content_hash"] = api_route_quote_lock_content_hash(raw)
    path = tmp_path / API_ROUTE_QUOTE_LOCK_FILENAME
    write_json(path, raw)

    with pytest.raises(ValueError, match="API route quote"):
        build_approval_bindings(
            config=config,
            preregistration=preregistration,
            gpu_lock=gpu_lock,
            quote_lock=quote,
            api_quote_lock=load_api_route_quote_lock(path),
        )


def test_quote_lock_tampering_and_extra_fields_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "gpu_quote_lock.json"
    write_json(path, _quote())
    tampered = read_json(path)
    tampered["usd_per_gpu_hour"] = 2.9
    write_json(path, tampered)
    with pytest.raises(GpuQuoteLockError, match="content hash"):
        load_gpu_quote_lock(path)

    extra = _quote()
    extra["secret"] = "should-not-be-here"
    extra["content_hash"] = gpu_quote_lock_content_hash(extra)
    write_json(path, extra)
    with pytest.raises(GpuQuoteLockError, match="schema"):
        load_gpu_quote_lock(path)


def test_quote_lock_rejects_non_eight_gpu_primary_offer(tmp_path: Path) -> None:
    path = tmp_path / "gpu_quote_lock.json"
    raw = _quote()
    raw["gpu_count"] = 4
    raw["content_hash"] = gpu_quote_lock_content_hash(raw)
    write_json(path, raw)
    with pytest.raises(GpuQuoteLockError, match="exactly eight"):
        load_gpu_quote_lock(path)


@pytest.mark.parametrize("operation", ["missing", "duplicate", "unknown"])
def test_gpu_quote_lock_requires_one_runtime_allocation_per_canonical_gpu_phase(
    tmp_path: Path, operation: str
) -> None:
    path = tmp_path / "gpu_quote_lock.json"
    raw = _quote()
    allocations = raw["phase_runtime_allocations"]
    if operation == "missing":
        allocations.pop()
    elif operation == "duplicate":
        allocations[-1] = dict(allocations[0])
    else:
        allocations[0]["command_phase"] = "custom_gpu"
    raw["content_hash"] = gpu_quote_lock_content_hash(raw)
    write_json(path, raw)

    with pytest.raises(GpuQuoteLockError, match="schema"):
        load_gpu_quote_lock(path)
