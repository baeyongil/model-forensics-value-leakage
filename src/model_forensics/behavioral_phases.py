"""Crash-safe GPU-only phases for behavioral rollout generation.

The expensive model is constructed only after an immutable request plan has
been written and validated.  Every deterministic microbatch is committed as an
atomic, content-authenticated JSONL artifact, so a restart skips completed work
and fails closed on drift or corruption.  This module deliberately imports no
judge, provider, embedding, or classification code; API adjudication belongs to
the subsequent CPU-only phase.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from model_forensics.io import (
    assert_unique,
    read_json,
    read_jsonl,
    sha256_file,
    stable_hash,
    write_json,
    write_jsonl,
)
from model_forensics.sampling import (
    GenerationBackend,
    GenerationRequest,
    materialize_rollout_rows,
)

BEHAVIORAL_GENERATION_PROTOCOL = "behavioral-gpu-generation-v2"
BEHAVIORAL_ENVIRONMENT_PROTOCOL = "behavioral-shared-generation-environment-v1"
_PHASES = frozenset({"baseline", "treatment"})
_REQUIRED_PRODUCTION_ENVIRONMENT_KEYS = frozenset(
    {
        "container_image_digest",
        "gpu_family",
        "gpu_count",
        "dtype",
        "tensor_parallel_size",
        "vllm_wheel_sha256",
    }
)


class BehavioralPhaseError(RuntimeError):
    """A frozen behavioral generation artifact is missing or inconsistent."""


@dataclass(frozen=True)
class BehavioralGenerationPhase:
    rows: tuple[dict[str, Any], ...]
    plan: Mapping[str, Any]
    manifest: Mapping[str, Any] | None
    complete: bool


def _without_hash(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key != field}


def _request_payload(request: GenerationRequest) -> dict[str, Any]:
    payload = asdict(request)
    payload["metadata"] = dict(request.metadata)
    return payload


def _normalize_execution_environment(
    value: Mapping[str, Any] | None,
    *,
    production: bool,
) -> dict[str, Any]:
    environment = {} if value is None else dict(value)
    if not production and not environment:
        return environment
    missing = sorted(_REQUIRED_PRODUCTION_ENVIRONMENT_KEYS - set(environment))
    if missing:
        raise BehavioralPhaseError(
            "production behavioral generation environment is missing " + ", ".join(missing)
        )
    container = environment.get("container_image_digest")
    if (
        not isinstance(container, str)
        or re.fullmatch(r"[^\s]+@sha256:[0-9a-f]{64}", container) is None
    ):
        raise BehavioralPhaseError("behavioral container image must be pinned by SHA-256")
    wheel = environment.get("vllm_wheel_sha256")
    if not isinstance(wheel, str) or re.fullmatch(r"[0-9a-f]{64}", wheel) is None:
        raise BehavioralPhaseError("behavioral vLLM wheel must have an exact SHA-256")
    for name in ("gpu_count", "tensor_parallel_size"):
        observed = environment.get(name)
        if isinstance(observed, bool) or not isinstance(observed, int) or observed <= 0:
            raise BehavioralPhaseError(f"behavioral {name} must be a positive integer")
    if environment["gpu_count"] != environment["tensor_parallel_size"]:
        raise BehavioralPhaseError(
            "behavioral GPU count must equal the frozen tensor-parallel topology"
        )
    for name in ("gpu_family", "dtype"):
        observed = environment.get(name)
        if not isinstance(observed, str) or not observed.strip():
            raise BehavioralPhaseError(f"behavioral {name} must be nonempty")
    return environment


def build_behavioral_generation_environment_identity(
    *,
    execution_environment: Mapping[str, Any],
    backend_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind phase-invariant approval/runtime identity to observed model provenance."""

    backend = dict(backend_provenance)
    if not backend:
        raise BehavioralPhaseError("observed backend provenance must be nonempty")
    production = backend.get("backend") == "vllm_offline"
    environment = _normalize_execution_environment(
        execution_environment,
        production=production,
    )
    if production:
        for name in ("dtype", "tensor_parallel_size"):
            if backend.get(name) != environment[name]:
                raise BehavioralPhaseError(
                    f"observed backend {name} disagrees with the approved environment"
                )
        required_backend = {
            "model_id",
            "model_revision",
            "tokenizer_id",
            "tokenizer_revision",
            "max_model_len",
            "chat_template_kwargs_hash",
            "detokenization_kwargs_hash",
            "chat_template_hash",
            "vllm_version",
            "transformers_version",
        }
        missing_backend = sorted(required_backend - set(backend))
        if missing_backend:
            raise BehavioralPhaseError(
                "production backend provenance is missing " + ", ".join(missing_backend)
            )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "protocol_version": BEHAVIORAL_ENVIRONMENT_PROTOCOL,
        "execution_environment": environment,
        "backend_provenance": backend,
    }
    payload["identity_hash"] = stable_hash(payload)
    return payload


def validate_behavioral_generation_environment_identity(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Authenticate a persisted shared behavioral generation identity."""

    identity = dict(value)
    if (
        identity.get("schema_version") != 1
        or identity.get("protocol_version") != BEHAVIORAL_ENVIRONMENT_PROTOCOL
        or identity.get("identity_hash") != stable_hash(_without_hash(identity, "identity_hash"))
    ):
        raise BehavioralPhaseError("behavioral generation environment identity hash mismatch")
    execution = identity.get("execution_environment")
    backend = identity.get("backend_provenance")
    if not isinstance(execution, Mapping) or not isinstance(backend, Mapping):
        raise BehavioralPhaseError("behavioral generation environment identity is malformed")
    expected = build_behavioral_generation_environment_identity(
        execution_environment=execution,
        backend_provenance=backend,
    )
    if expected != identity:
        raise BehavioralPhaseError("behavioral generation environment identity is noncanonical")
    return identity


def _validate_phase_requests(
    requests: Sequence[GenerationRequest],
    *,
    phase: str,
) -> None:
    if phase not in _PHASES:
        raise BehavioralPhaseError(f"phase must be one of {sorted(_PHASES)}")
    if not requests:
        raise BehavioralPhaseError("generation phase requires at least one request")
    request_ids = [request.request_id for request in requests]
    if len(request_ids) != len(set(request_ids)):
        raise BehavioralPhaseError("generation request IDs must be unique")
    conditions = {request.condition for request in requests}
    if phase == "baseline" and conditions != {"baseline"}:
        raise BehavioralPhaseError("baseline phase may contain only baseline requests")
    if phase == "treatment" and "baseline" in conditions:
        raise BehavioralPhaseError("treatment phase must not contain baseline requests")


def _plan_payload(
    *,
    requests: Sequence[GenerationRequest],
    phase: str,
    batch_size: int,
    expected_backend_provenance: Mapping[str, Any],
    expected_execution_environment: Mapping[str, Any] | None,
) -> dict[str, Any]:
    _validate_phase_requests(requests, phase=phase)
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise BehavioralPhaseError("batch_size must be a positive integer")
    if not expected_backend_provenance:
        raise BehavioralPhaseError("expected backend provenance must be frozen")
    production = expected_backend_provenance.get("backend") == "vllm_offline"
    execution_environment = _normalize_execution_environment(
        expected_execution_environment,
        production=production,
    )
    request_rows = [_request_payload(request) for request in requests]
    batch_count = (len(request_rows) + batch_size - 1) // batch_size
    payload: dict[str, Any] = {
        "schema_version": 1,
        "protocol_version": BEHAVIORAL_GENERATION_PROTOCOL,
        "phase": phase,
        "request_count": len(request_rows),
        "request_manifest_hash": stable_hash(request_rows),
        "request_ids": [request.request_id for request in requests],
        "batch_size": batch_size,
        "batch_count": batch_count,
        "expected_backend_provenance": dict(expected_backend_provenance),
        "expected_backend_provenance_hash": stable_hash(dict(expected_backend_provenance)),
        "expected_execution_environment": execution_environment,
        "expected_execution_environment_hash": stable_hash(execution_environment),
        "requests": request_rows,
    }
    payload["plan_hash"] = stable_hash(payload)
    return payload


def freeze_behavioral_generation_plan(
    *,
    requests: Sequence[GenerationRequest],
    phase: str,
    checkpoint_dir: str | Path,
    batch_size: int,
    expected_backend_provenance: Mapping[str, Any],
    expected_execution_environment: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write or verify the exact request plan before model construction."""

    directory = Path(checkpoint_dir)
    plan = _plan_payload(
        requests=requests,
        phase=phase,
        batch_size=batch_size,
        expected_backend_provenance=expected_backend_provenance,
        expected_execution_environment=expected_execution_environment,
    )
    path = directory / "generation_plan.json"
    if path.exists():
        observed = read_json(path)
        # JSON round-tripping normalizes tuple-valued dataclass fields (for
        # example stop markers) to arrays, so compare canonical content hashes.
        if not isinstance(observed, dict) or stable_hash(observed) != stable_hash(plan):
            raise BehavioralPhaseError("frozen generation plan mismatch")
        return observed
    write_json(path, plan)
    return plan


def _provenance_matches(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    return all(key in actual and actual[key] == value for key, value in expected.items())


def _batch_path(directory: Path, index: int) -> Path:
    return directory / "batches" / f"batch-{index:05d}.jsonl"


def _expected_batch_requests(
    requests: Sequence[GenerationRequest], *, batch_size: int, index: int
) -> Sequence[GenerationRequest]:
    start = index * batch_size
    return requests[start : start + batch_size]


def _validate_batch_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    requests: Sequence[GenerationRequest],
    phase: str,
    batch_index: int,
    dispatch_start: int,
    plan_hash: str,
    expected_backend_provenance: Mapping[str, Any],
) -> list[dict[str, Any]]:
    expected_ids = [request.request_id for request in requests]
    observed_ids = [str(row.get("run_id", "")) for row in rows]
    if observed_ids != expected_ids:
        raise BehavioralPhaseError(f"batch {batch_index} request inventory or order mismatch")
    normalized: list[dict[str, Any]] = []
    for offset, source in enumerate(rows):
        row = dict(source)
        recorded_hash = row.get("record_hash")
        if recorded_hash != stable_hash(_without_hash(row, "record_hash")):
            raise BehavioralPhaseError(f"batch {batch_index} record hash mismatch")
        expected_dispatch = dispatch_start + offset
        if row.get("sampling_phase") != phase:
            raise BehavioralPhaseError(f"batch {batch_index} phase mismatch")
        if row.get("generation_batch_index") != batch_index:
            raise BehavioralPhaseError(f"batch {batch_index} index mismatch")
        if row.get("dispatch_order") != expected_dispatch:
            raise BehavioralPhaseError(f"batch {batch_index} dispatch order mismatch")
        if row.get("generation_plan_hash") != plan_hash:
            raise BehavioralPhaseError(f"batch {batch_index} plan hash mismatch")
        backend = row.get("backend")
        if not isinstance(backend, Mapping) or not _provenance_matches(
            backend, expected_backend_provenance
        ):
            raise BehavioralPhaseError(f"batch {batch_index} backend provenance mismatch")
        normalized.append(row)
    assert_unique(normalized, "run_id")
    return normalized


def _load_existing_batches(
    *,
    directory: Path,
    plan: Mapping[str, Any],
    requests: Sequence[GenerationRequest],
) -> tuple[list[dict[str, Any]], int]:
    batch_size = int(plan["batch_size"])
    batch_count = int(plan["batch_count"])
    rows: list[dict[str, Any]] = []
    first_missing = batch_count
    missing_seen = False
    for index in range(batch_count):
        path = _batch_path(directory, index)
        if not path.exists():
            if not missing_seen:
                first_missing = index
                missing_seen = True
            continue
        if missing_seen:
            raise BehavioralPhaseError(
                "non-contiguous batch checkpoints; refusing ambiguous resume"
            )
        batch_requests = _expected_batch_requests(requests, batch_size=batch_size, index=index)
        batch_rows = _validate_batch_rows(
            read_jsonl(path),
            requests=batch_requests,
            phase=str(plan["phase"]),
            batch_index=index,
            dispatch_start=index * batch_size,
            plan_hash=str(plan["plan_hash"]),
            expected_backend_provenance=plan["expected_backend_provenance"],
        )
        rows.extend(batch_rows)
    assert_unique(rows, "run_id")
    return rows, first_missing


def _finalize_generation_phase(
    *,
    directory: Path,
    plan: Mapping[str, Any],
    requests: Sequence[GenerationRequest],
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(rows) != len(requests):
        raise BehavioralPhaseError("cannot finalize an incomplete generation phase")
    expected_ids = [request.request_id for request in requests]
    if [str(row["run_id"]) for row in rows] != expected_ids:
        raise BehavioralPhaseError("final generation row order disagrees with frozen requests")
    backend_provenances = [row.get("backend") for row in rows]
    if not backend_provenances or not isinstance(backend_provenances[0], Mapping):
        raise BehavioralPhaseError("completed generation lacks backend provenance")
    observed_backend = dict(backend_provenances[0])
    if any(
        not isinstance(item, Mapping) or dict(item) != observed_backend
        for item in backend_provenances
    ):
        raise BehavioralPhaseError(
            "completed generation mixes different backend environment identities"
        )
    shared_environment = build_behavioral_generation_environment_identity(
        execution_environment=plan.get("expected_execution_environment", {}),
        backend_provenance=observed_backend,
    )
    merged_path = directory / "generation_rows.jsonl"
    if merged_path.exists():
        observed = read_jsonl(merged_path)
        if observed != list(rows):
            raise BehavioralPhaseError("existing merged generation rows mismatch")
    else:
        write_jsonl(merged_path, rows)
    batch_files: list[dict[str, Any]] = []
    for index in range(int(plan["batch_count"])):
        path = _batch_path(directory, index)
        if not path.exists():
            raise BehavioralPhaseError(f"missing completed batch {index}")
        batch_rows = read_jsonl(path)
        batch_files.append(
            {
                "batch_index": index,
                "path": f"batches/{path.name}",
                "row_count": len(batch_rows),
                "sha256": sha256_file(path),
                "row_hashes_hash": stable_hash([row.get("record_hash") for row in batch_rows]),
            }
        )
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "protocol_version": BEHAVIORAL_GENERATION_PROTOCOL,
        "phase": plan["phase"],
        "plan_hash": plan["plan_hash"],
        "complete": True,
        "request_count": len(requests),
        "row_count": len(rows),
        "run_ids_hash": stable_hash([row["run_id"] for row in rows]),
        "row_hashes_hash": stable_hash([row["record_hash"] for row in rows]),
        "observed_backend_provenance": observed_backend,
        "observed_backend_provenance_hash": stable_hash(observed_backend),
        "shared_generation_environment": shared_environment,
        "shared_generation_environment_hash": shared_environment["identity_hash"],
        "merged_rows_path": merged_path.name,
        "merged_rows_sha256": sha256_file(merged_path),
        "batch_files": batch_files,
    }
    manifest["manifest_hash"] = stable_hash(manifest)
    manifest_path = directory / "generation_manifest.json"
    if manifest_path.exists():
        observed_manifest = read_json(manifest_path)
        if observed_manifest != manifest:
            raise BehavioralPhaseError("existing generation manifest mismatch")
    else:
        write_json(manifest_path, manifest)
    return manifest


def run_behavioral_generation_phase(
    *,
    requests: Sequence[GenerationRequest],
    backend_factory: Callable[[], GenerationBackend],
    phase: str,
    checkpoint_dir: str | Path,
    batch_size: int,
    expected_backend_provenance: Mapping[str, Any],
    expected_execution_environment: Mapping[str, Any] | None = None,
    max_new_batches: int | None = None,
    on_batch_committed: Callable[[Path, Sequence[Mapping[str, Any]]], None] | None = None,
    before_backend: Callable[[Mapping[str, Any]], None] | None = None,
) -> BehavioralGenerationPhase:
    """Run one GPU-only phase with immutable, resumable microbatch commits."""

    if max_new_batches is not None and (
        isinstance(max_new_batches, bool)
        or not isinstance(max_new_batches, int)
        or max_new_batches < 0
    ):
        raise BehavioralPhaseError("max_new_batches must be a non-negative integer or None")
    directory = Path(checkpoint_dir)
    plan = freeze_behavioral_generation_plan(
        requests=requests,
        phase=phase,
        checkpoint_dir=directory,
        batch_size=batch_size,
        expected_backend_provenance=expected_backend_provenance,
        expected_execution_environment=expected_execution_environment,
    )
    rows, first_missing = _load_existing_batches(
        directory=directory,
        plan=plan,
        requests=requests,
    )
    batch_count = int(plan["batch_count"])
    if first_missing == batch_count:
        manifest = _finalize_generation_phase(
            directory=directory,
            plan=plan,
            requests=requests,
            rows=rows,
        )
        return BehavioralGenerationPhase(tuple(rows), plan, manifest, True)
    if max_new_batches == 0:
        return BehavioralGenerationPhase(tuple(rows), plan, None, False)

    if before_backend is not None:
        before_backend(plan)
    backend = backend_factory()
    actual_provenance = dict(backend.provenance)
    if not _provenance_matches(actual_provenance, expected_backend_provenance):
        raise BehavioralPhaseError("constructed backend disagrees with frozen provenance")
    if rows:
        previous = rows[0].get("backend")
        if not isinstance(previous, Mapping) or dict(previous) != actual_provenance:
            raise BehavioralPhaseError(
                "constructed backend environment differs from completed batch checkpoints"
            )
    build_behavioral_generation_environment_identity(
        execution_environment=plan.get("expected_execution_environment", {}),
        backend_provenance=actual_provenance,
    )
    generated_batches = 0
    batch_size_value = int(plan["batch_size"])
    for index in range(first_missing, batch_count):
        if max_new_batches is not None and generated_batches >= max_new_batches:
            break
        batch_requests = _expected_batch_requests(
            requests, batch_size=batch_size_value, index=index
        )
        results = backend.generate(batch_requests)
        batch_rows = materialize_rollout_rows(
            batch_requests,
            results,
            backend_provenance=actual_provenance,
        )
        for offset, row in enumerate(batch_rows):
            row.pop("record_hash", None)
            row.update(
                {
                    "sampling_phase": phase,
                    "generation_batch_index": index,
                    "dispatch_order": index * batch_size_value + offset,
                    "generation_plan_hash": plan["plan_hash"],
                }
            )
            row["record_hash"] = stable_hash(row)
        validated = _validate_batch_rows(
            batch_rows,
            requests=batch_requests,
            phase=phase,
            batch_index=index,
            dispatch_start=index * batch_size_value,
            plan_hash=str(plan["plan_hash"]),
            expected_backend_provenance=expected_backend_provenance,
        )
        path = _batch_path(directory, index)
        if path.exists():  # defensive: another writer would make resume ambiguous
            raise BehavioralPhaseError(f"batch checkpoint unexpectedly appeared: {path}")
        write_jsonl(path, validated)
        if on_batch_committed is not None:
            on_batch_committed(path, tuple(validated))
        rows.extend(validated)
        generated_batches += 1

    complete = len(rows) == len(requests)
    manifest = None
    if complete:
        manifest = _finalize_generation_phase(
            directory=directory,
            plan=plan,
            requests=requests,
            rows=rows,
        )
    return BehavioralGenerationPhase(tuple(rows), plan, manifest, complete)


def load_behavioral_generation_phase(
    checkpoint_dir: str | Path,
) -> BehavioralGenerationPhase:
    """Load and fully authenticate a completed GPU-generation phase."""

    directory = Path(checkpoint_dir)
    plan_path = directory / "generation_plan.json"
    manifest_path = directory / "generation_manifest.json"
    if not plan_path.exists() or not manifest_path.exists():
        raise BehavioralPhaseError("completed generation plan and manifest are required")
    plan = read_json(plan_path)
    manifest = read_json(manifest_path)
    if not isinstance(plan, dict) or plan.get("plan_hash") != stable_hash(
        _without_hash(plan, "plan_hash")
    ):
        raise BehavioralPhaseError("generation plan hash mismatch")
    if not isinstance(manifest, dict) or manifest.get("manifest_hash") != stable_hash(
        _without_hash(manifest, "manifest_hash")
    ):
        raise BehavioralPhaseError("generation manifest hash mismatch")
    if manifest.get("plan_hash") != plan.get("plan_hash"):
        raise BehavioralPhaseError("generation manifest references a different plan")
    expected_environment = plan.get("expected_execution_environment")
    if not isinstance(expected_environment, Mapping) or plan.get(
        "expected_execution_environment_hash"
    ) != stable_hash(dict(expected_environment)):
        raise BehavioralPhaseError("generation plan execution environment hash mismatch")
    shared_environment = manifest.get("shared_generation_environment")
    if not isinstance(shared_environment, Mapping):
        raise BehavioralPhaseError("generation manifest lacks shared environment identity")
    authenticated_environment = validate_behavioral_generation_environment_identity(
        shared_environment
    )
    if manifest.get("shared_generation_environment_hash") != authenticated_environment.get(
        "identity_hash"
    ):
        raise BehavioralPhaseError("generation manifest shared environment hash mismatch")
    if authenticated_environment.get("execution_environment") != dict(expected_environment):
        raise BehavioralPhaseError("generation environment disagrees with its frozen plan")
    batch_files = manifest.get("batch_files")
    if not isinstance(batch_files, list) or [
        item.get("batch_index") for item in batch_files if isinstance(item, Mapping)
    ] != list(range(int(plan["batch_count"]))):
        raise BehavioralPhaseError("generation manifest batch inventory mismatch")
    request_rows = plan.get("requests")
    if not isinstance(request_rows, list):
        raise BehavioralPhaseError("generation plan request inventory is malformed")

    # Rehydrate only the fields needed for strict batch identity checks.  Exact
    # generated rows are loaded from the authenticated batch files below.
    request_ids = plan.get("request_ids")
    if not isinstance(request_ids, list) or len(request_ids) != int(plan["request_count"]):
        raise BehavioralPhaseError("generation plan request IDs are malformed")
    rows: list[dict[str, Any]] = []
    for item in batch_files:
        if not isinstance(item, Mapping):
            raise BehavioralPhaseError("generation manifest batch entry is malformed")
        relative = item.get("path")
        if (
            not isinstance(relative, str)
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
        ):
            raise BehavioralPhaseError("generation manifest contains an unsafe batch path")
        path = directory / relative
        if not path.exists() or sha256_file(path) != item.get("sha256"):
            raise BehavioralPhaseError("generation batch file hash mismatch")
        batch_rows = read_jsonl(path)
        if len(batch_rows) != item.get("row_count"):
            raise BehavioralPhaseError("generation batch row count mismatch")
        if stable_hash([row.get("record_hash") for row in batch_rows]) != item.get(
            "row_hashes_hash"
        ):
            raise BehavioralPhaseError("generation batch row hash inventory mismatch")
        for row in batch_rows:
            if row.get("record_hash") != stable_hash(_without_hash(row, "record_hash")):
                raise BehavioralPhaseError("generation batch record hash mismatch")
        rows.extend(batch_rows)
    if [row.get("run_id") for row in rows] != request_ids:
        raise BehavioralPhaseError("generation batch inventory disagrees with plan")
    merged_path = directory / str(manifest.get("merged_rows_path"))
    if not merged_path.exists() or sha256_file(merged_path) != manifest.get("merged_rows_sha256"):
        raise BehavioralPhaseError("merged generation rows file hash mismatch")
    if read_jsonl(merged_path) != rows:
        raise BehavioralPhaseError("merged generation rows disagree with batch checkpoints")
    if len(rows) != manifest.get("row_count") or len(rows) != plan.get("request_count"):
        raise BehavioralPhaseError("completed generation row count mismatch")
    if stable_hash([row["record_hash"] for row in rows]) != manifest.get("row_hashes_hash"):
        raise BehavioralPhaseError("completed generation row hash inventory mismatch")
    observed_backend = manifest.get("observed_backend_provenance")
    if (
        not isinstance(observed_backend, Mapping)
        or manifest.get("observed_backend_provenance_hash") != stable_hash(dict(observed_backend))
        or authenticated_environment.get("backend_provenance") != dict(observed_backend)
        or any(row.get("backend") != observed_backend for row in rows)
    ):
        raise BehavioralPhaseError("completed generation backend identity mismatch")
    assert_unique(rows, "run_id")
    return BehavioralGenerationPhase(tuple(rows), plan, manifest, True)


__all__ = [
    "BEHAVIORAL_ENVIRONMENT_PROTOCOL",
    "BEHAVIORAL_GENERATION_PROTOCOL",
    "BehavioralGenerationPhase",
    "BehavioralPhaseError",
    "build_behavioral_generation_environment_identity",
    "freeze_behavioral_generation_plan",
    "load_behavioral_generation_phase",
    "run_behavioral_generation_phase",
    "validate_behavioral_generation_environment_identity",
]
