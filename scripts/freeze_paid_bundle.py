#!/usr/bin/env python3
"""Freeze and validate the private quote/approval bundle without paid work.

``preview`` authenticates existing quote locks, or exclusively creates them
from two explicit unhashed JSON specs, then prints a secret-free cost summary.
``approve`` must be invoked separately after explicit user approval; it never
infers an approval identifier, timestamp, or command phase.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import os
import re
import stat
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import ROUND_FLOOR, Decimal
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from model_forensics.approval import (
    APPROVAL_FILENAME,
    APPROVAL_SCHEMA_VERSION,
    MAX_API_QUOTE_AGE,
    MAX_FUTURE_CLOCK_SKEW,
    MAX_GPU_QUOTE_AGE,
    PAID_COMMAND_PHASES,
    PAID_RUN_REVIEW_PROTOCOL_VERSION,
    ApprovalBindings,
    PaidRunApproval,
    PaidRunApprovalError,
    PaidRunReview,
    PaidRunReviewPayload,
    UserApproval,
    approval_content_hash,
    canonicalize_paid_command_phases,
    load_paid_run_approval,
    paid_run_review_hash,
    require_clean_source_commit,
    validate_paid_run_approval,
)
from model_forensics.budget import BudgetLimits, CostLedger
from model_forensics.config import RunConfig
from model_forensics.execution_bindings import (
    API_ROUTE_QUOTE_LOCK_FILENAME,
    GPU_QUOTE_LOCK_FILENAME,
    ApiRouteQuoteLock,
    GpuQuoteLock,
    api_route_quote_lock_content_hash,
    build_approval_bindings,
    gpu_quote_lock_content_hash,
)
from model_forensics.gpu_budget import (
    approved_gpu_phase_maximum_usd,
    write_json_exclusive,
)
from model_forensics.io import stable_hash
from model_forensics.paid_bundle_rotation import PaidBundleRotationError, paid_bundle_lock


class PaidBundleError(ValueError):
    """A private paid-run input is unsafe, malformed, or inconsistent."""


class _UniqueSafeLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(
    loader: _UniqueSafeLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in result
        except TypeError as exc:
            raise PaidBundleError("YAML mapping contains an unhashable key") from exc
        if duplicate:
            raise PaidBundleError("YAML mapping contains a duplicate key")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueSafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _stable_regular_file_bytes(path: Path, *, project_root: Path, label: str) -> bytes:
    """Read one project file through a stable non-following descriptor."""

    try:
        relative = path.relative_to(project_root)
    except ValueError as exc:
        raise PaidBundleError(f"{label} must remain inside the project") from exc
    current = project_root
    for component in relative.parts[:-1]:
        current = current / component
        try:
            details = current.lstat()
        except OSError as exc:
            raise PaidBundleError(f"{label} parent directory is unavailable") from exc
        if stat.S_ISLNK(details.st_mode) or not stat.S_ISDIR(details.st_mode):
            raise PaidBundleError(f"{label} path must not contain a symlink")

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise PaidBundleError(f"{label} must be a readable regular non-symlink file") from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_uid != os.getuid()
        ):
            raise PaidBundleError(f"{label} owner, type, or link count is unsafe")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        raw = b"".join(chunks)
        if identity_before != identity_after or len(raw) != after.st_size:
            raise PaidBundleError(f"{label} changed while it was being read")
        try:
            current_path = path.lstat()
        except OSError as exc:
            raise PaidBundleError(f"{label} changed while it was being read") from exc
        if (
            current_path.st_dev != after.st_dev
            or current_path.st_ino != after.st_ino
            or stat.S_ISLNK(current_path.st_mode)
        ):
            raise PaidBundleError(f"{label} path changed while it was being read")
        return raw
    finally:
        os.close(descriptor)


def _load_yaml_mapping(
    path: Path,
    *,
    project_root: Path,
    label: str,
) -> dict[str, Any]:
    raw = _stable_regular_file_bytes(path, project_root=project_root, label=label)
    value = yaml.load(raw.decode("utf-8"), Loader=_UniqueSafeLoader)
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise PaidBundleError(f"{label} must be a string-keyed YAML mapping")
    return value


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PaidBundleError("quote spec contains a duplicate JSON key")
        result[key] = value
    return result


def _load_json_mapping(
    path: Path,
    *,
    project_root: Path,
    label: str,
) -> dict[str, Any]:
    raw_bytes = _stable_regular_file_bytes(path, project_root=project_root, label=label)
    value = json.loads(
        raw_bytes.decode("utf-8"),
        object_pairs_hook=_unique_json_object,
    )
    if not isinstance(value, dict):
        raise PaidBundleError(f"{label} must be a JSON object")
    return value


def _load_unhashed_quote_spec(
    path: Path,
    *,
    project_root: Path,
    label: str,
) -> dict[str, Any]:
    _secure_existing_private_file(path, label=label)
    value = _load_json_mapping(
        path,
        project_root=project_root,
        label=label,
    )
    if "content_hash" in value:
        raise PaidBundleError(f"{label} must omit content_hash so it is computed automatically")
    return value


def _absolute(path: Path, *, base: Path) -> Path:
    candidate = path if path.is_absolute() else base / path
    return Path(os.path.abspath(os.fspath(candidate)))


def _assert_no_symlink_chain(path: Path, *, private_root: Path) -> None:
    try:
        relative = path.relative_to(private_root)
    except ValueError as exc:
        raise PaidBundleError(f"private artifact must remain under {private_root}: {path}") from exc
    current = private_root
    for component in relative.parts:
        current = current / component
        if os.path.lexists(current) and current.is_symlink():
            raise PaidBundleError(f"private artifact path must not contain a symlink: {current}")


def _prepare_private_root(project_root: Path) -> Path:
    private_root = project_root / ".runpod"
    if os.path.lexists(private_root) and private_root.is_symlink():
        raise PaidBundleError(f"private .runpod root must not be a symlink: {private_root}")
    private_root.mkdir(mode=0o700, exist_ok=True)
    if not private_root.is_dir():
        raise PaidBundleError(f"private .runpod root is not a directory: {private_root}")
    private_root.chmod(0o700)
    return private_root


def _private_path(raw: Path, *, project_root: Path, create_parent: bool) -> Path:
    private_root = _prepare_private_root(project_root)
    path = _absolute(raw, base=project_root)
    _assert_no_symlink_chain(path, private_root=private_root)
    if create_parent:
        current = private_root
        for component in path.parent.relative_to(private_root).parts:
            current = current / component
            if os.path.lexists(current) and current.is_symlink():
                raise PaidBundleError(f"private directory must not be a symlink: {current}")
            current.mkdir(mode=0o700, exist_ok=True)
            if not current.is_dir():
                raise PaidBundleError(f"private path component is not a directory: {current}")
            current.chmod(0o700)
    return path


def _secure_existing_private_file(path: Path, *, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise PaidBundleError(f"{label} must be a regular non-symlink file: {path}")
    mode = path.lstat().st_mode
    if not stat.S_ISREG(mode):
        raise PaidBundleError(f"{label} must be a regular file: {path}")
    if path.stat().st_nlink != 1:
        raise PaidBundleError(f"{label} must not be hard-linked: {path}")
    path.chmod(0o600)


def _freeze_quote_specs(
    *,
    project_root: Path,
    gpu_spec_path: Path,
    api_spec_path: Path,
    gpu_quote_path: Path,
    api_quote_path: Path,
) -> None:
    gpu_raw = _load_unhashed_quote_spec(
        gpu_spec_path,
        project_root=project_root,
        label="GPU quote spec",
    )
    api_raw = _load_unhashed_quote_spec(
        api_spec_path,
        project_root=project_root,
        label="API quote spec",
    )
    gpu_payload = {**gpu_raw, "content_hash": gpu_quote_lock_content_hash(gpu_raw)}
    api_payload = {**api_raw, "content_hash": api_route_quote_lock_content_hash(api_raw)}

    # Validate both payloads before either canonical destination is claimed.
    GpuQuoteLock.model_validate(gpu_payload)
    ApiRouteQuoteLock.model_validate(api_payload)
    for destination in (gpu_quote_path, api_quote_path):
        if os.path.lexists(destination):
            raise PaidBundleError(f"refusing to overwrite claimed quote lock: {destination}")
    write_json_exclusive(gpu_quote_path, gpu_payload)
    write_json_exclusive(api_quote_path, api_payload)


@dataclass(frozen=True)
class BundleContext:
    project_root: Path
    config_path: Path
    preregistration_path: Path
    gpu_lock_path: Path
    gpu_quote_path: Path
    api_quote_path: Path
    config: RunConfig
    preregistration: Mapping[str, Any]
    gpu_lock: Mapping[str, Any]
    gpu_quote: GpuQuoteLock
    api_quote: ApiRouteQuoteLock
    bindings: ApprovalBindings


def _load_context(args: argparse.Namespace, *, allow_quote_specs: bool) -> BundleContext:
    config_path = _absolute(Path(args.config), base=Path.cwd())
    project_root = config_path.parent.parent
    config_raw = _load_yaml_mapping(
        config_path,
        project_root=project_root,
        label="run config",
    )
    config = RunConfig.model_validate(config_raw)
    config.source_path = config_path
    config.assert_execution_ready()

    preregistration_path = _absolute(Path(args.preregistration), base=Path.cwd())
    configured_preregistration = config.preregistration
    if not configured_preregistration.is_absolute():
        configured_preregistration = project_root / configured_preregistration
    configured_preregistration = _absolute(
        configured_preregistration,
        base=project_root,
    )
    if preregistration_path != configured_preregistration:
        raise PaidBundleError(
            "explicit preregistration path disagrees with the run config: "
            f"{preregistration_path} != {configured_preregistration}"
        )
    preregistration = _load_yaml_mapping(
        preregistration_path,
        project_root=project_root,
        label="preregistration",
    )
    gpu_lock_path = _absolute(Path(args.gpu_lock), base=Path.cwd())
    gpu_lock = _load_yaml_mapping(
        gpu_lock_path,
        project_root=project_root,
        label="GPU/software lock",
    )

    gpu_quote_path = _private_path(
        Path(args.gpu_quote_lock),
        project_root=project_root,
        create_parent=True,
    )
    api_quote_path = _private_path(
        Path(args.api_quote_lock),
        project_root=project_root,
        create_parent=True,
    )
    if gpu_quote_path.name != GPU_QUOTE_LOCK_FILENAME:
        raise PaidBundleError(f"GPU quote lock must be named {GPU_QUOTE_LOCK_FILENAME}")
    if api_quote_path.name != API_ROUTE_QUOTE_LOCK_FILENAME:
        raise PaidBundleError(f"API quote lock must be named {API_ROUTE_QUOTE_LOCK_FILENAME}")

    gpu_spec = getattr(args, "gpu_quote_spec", None)
    api_spec = getattr(args, "api_quote_spec", None)
    if bool(gpu_spec) != bool(api_spec):
        raise PaidBundleError("both --gpu-quote-spec and --api-quote-spec are required together")
    if gpu_spec:
        if not allow_quote_specs:
            raise PaidBundleError("quote specs are accepted only by preview")
        gpu_spec_path = _private_path(
            Path(gpu_spec),
            project_root=project_root,
            create_parent=False,
        )
        api_spec_path = _private_path(
            Path(api_spec),
            project_root=project_root,
            create_parent=False,
        )
        _freeze_quote_specs(
            project_root=project_root,
            gpu_spec_path=gpu_spec_path,
            api_spec_path=api_spec_path,
            gpu_quote_path=gpu_quote_path,
            api_quote_path=api_quote_path,
        )

    _secure_existing_private_file(gpu_quote_path, label="GPU quote lock")
    _secure_existing_private_file(api_quote_path, label="API quote lock")
    gpu_quote_raw = _load_json_mapping(
        gpu_quote_path,
        project_root=project_root,
        label="GPU quote lock",
    )
    if gpu_quote_raw.get("content_hash") != gpu_quote_lock_content_hash(gpu_quote_raw):
        raise PaidBundleError("GPU quote lock content hash mismatch")
    gpu_quote = GpuQuoteLock.model_validate(gpu_quote_raw)
    api_quote_raw = _load_json_mapping(
        api_quote_path,
        project_root=project_root,
        label="API quote lock",
    )
    if api_quote_raw.get("content_hash") != api_route_quote_lock_content_hash(api_quote_raw):
        raise PaidBundleError("API route quote lock content hash mismatch")
    api_quote = ApiRouteQuoteLock.model_validate(api_quote_raw)
    bindings = build_approval_bindings(
        config=config,
        preregistration=preregistration,
        gpu_lock=gpu_lock,
        quote_lock=gpu_quote,
        api_quote_lock=api_quote,
    )
    return BundleContext(
        project_root=project_root,
        config_path=config_path,
        preregistration_path=preregistration_path,
        gpu_lock_path=gpu_lock_path,
        gpu_quote_path=gpu_quote_path,
        api_quote_path=api_quote_path,
        config=config,
        preregistration=preregistration,
        gpu_lock=gpu_lock,
        gpu_quote=gpu_quote,
        api_quote=api_quote,
        bindings=bindings,
    )


def _timestamp_status(timestamp: datetime, *, maximum_age_seconds: float) -> dict[str, Any]:
    now = datetime.now(UTC)
    age_seconds = (now - timestamp.astimezone(UTC)).total_seconds()
    return {
        "timestamp": timestamp.astimezone(UTC).isoformat().replace("+00:00", "Z"),
        "age_seconds": round(age_seconds, 3),
        "fresh": -MAX_FUTURE_CLOCK_SKEW.total_seconds() <= age_seconds <= maximum_age_seconds,
    }


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


@dataclass(frozen=True)
class LedgerSnapshot:
    path: Path
    raw_bytes: bytes
    document: dict[str, Any]
    bytes_sha256: str
    document_hash: str


@dataclass(frozen=True)
class ReviewComputation:
    review: PaidRunReview
    planned_command_phases: tuple[str, ...]
    phase_maxima: tuple[dict[str, Any], ...]
    projected_compute_usd: float
    projected_running_storage_usd: float
    projected_gpu_usd: float
    cumulative: dict[str, Any]
    gpu_quote_status: dict[str, Any]
    api_quote_status: dict[str, Any]

    @property
    def ready(self) -> bool:
        needs_gpu_quote = any(phase.endswith("_gpu") for phase in self.planned_command_phases)
        needs_api_quote = any(phase.endswith("_api") for phase in self.planned_command_phases)
        return bool(
            (not needs_gpu_quote or self.gpu_quote_status["fresh"])
            and (not needs_api_quote or self.api_quote_status["fresh"])
        )


def _clean_source_commit(project_root: Path, *, ledger_path: Path) -> str:
    """Return HEAD only for a clean checkout plus the bound mutable ledger."""

    try:
        return require_clean_source_commit(
            project_root,
            mutable_paths=(ledger_path,),
        )
    except PaidRunApprovalError as exc:
        raise PaidBundleError(str(exc)) from exc


def _ledger_snapshot(
    context: BundleContext,
    *,
    cost_ledger_path: Path,
) -> LedgerSnapshot:
    ledger_path = _absolute(cost_ledger_path, base=context.project_root)
    canonical_ledger_path = _absolute(
        Path(context.config.paths.manifest_dir) / "cost_ledger.yaml",
        base=context.project_root,
    )
    if ledger_path != canonical_ledger_path:
        raise PaidBundleError("cost ledger must be the run config's canonical cumulative ledger")
    try:
        ledger_path.relative_to(context.project_root)
    except ValueError as exc:
        raise PaidBundleError("cost ledger must remain inside the project") from exc
    ledger = CostLedger(
        ledger_path,
        BudgetLimits(
            gpu=context.bindings.caps_usd.gpu,
            api=context.bindings.caps_usd.api,
            total=context.bindings.caps_usd.total,
        ),
    )
    try:
        with ledger._locked():  # type: ignore[attr-defined]
            before = _stable_regular_file_bytes(
                ledger_path,
                project_root=context.project_root,
                label="canonical cost ledger",
            )
            document = ledger._load_unlocked()  # type: ignore[attr-defined]
            after = _stable_regular_file_bytes(
                ledger_path,
                project_root=context.project_root,
                label="canonical cost ledger",
            )
    except (OSError, TypeError, ValueError) as exc:
        raise PaidBundleError("canonical cost ledger is invalid") from exc
    if before != after:
        raise PaidBundleError("canonical cost ledger changed during review")
    try:
        encoded_document = yaml.load(
            before.decode("utf-8"),
            Loader=_UniqueSafeLoader,
        )
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise PaidBundleError("canonical cost ledger bytes are invalid") from exc
    if encoded_document != document:
        raise PaidBundleError("canonical cost ledger bytes disagree with the validated document")
    return LedgerSnapshot(
        path=ledger_path,
        raw_bytes=before,
        document=document,
        bytes_sha256="sha256:" + hashlib.sha256(before).hexdigest(),
        document_hash=stable_hash(document),
    )


def _review_computation(
    context: BundleContext,
    *,
    cost_ledger_path: Path,
    gpu_safety_margin_fraction: float,
    planned_command_phases: tuple[str, ...],
) -> ReviewComputation:
    if (
        not math.isfinite(gpu_safety_margin_fraction)
        or not 0 < gpu_safety_margin_fraction < 0.25
    ):
        raise PaidBundleError("GPU safety margin must be in (0, 0.25)")
    ledger_path = _absolute(cost_ledger_path, base=context.project_root)
    source_commit_before = _clean_source_commit(
        context.project_root,
        ledger_path=ledger_path,
    )
    ledger = _ledger_snapshot(context, cost_ledger_path=ledger_path)
    bindings = context.bindings
    planned_command_phases = canonicalize_paid_command_phases(planned_command_phases)
    planned_phase_set = frozenset(planned_command_phases)
    allocations = [item.model_dump(mode="json") for item in bindings.gpu.phase_runtime_allocations]
    planned_gpu_allocations = [
        item for item in allocations if str(item["command_phase"]) in planned_phase_set
    ]
    allocated_hours = sum(
        float(item["maximum_runtime_hours"]) for item in planned_gpu_allocations
    )
    projected_compute_usd = (
        bindings.gpu.count * bindings.gpu.quote.usd_per_gpu_hour * allocated_hours
    )
    projected_running_storage_usd = (
        bindings.gpu.quote.running_storage_usd_per_hour * allocated_hours
    )
    phase_maxima = tuple(
        {
            "command_phase": str(item["command_phase"]),
            "maximum_usd": approved_gpu_phase_maximum_usd(
                gpu_count=bindings.gpu.count,
                quote_hourly_per_gpu_usd=bindings.gpu.quote.usd_per_gpu_hour,
                running_storage_hourly_usd=(
                    bindings.gpu.quote.running_storage_usd_per_hour
                ),
                approved_runtime_hours=float(item["maximum_runtime_hours"]),
            ),
        }
        for item in allocations
    )
    projected_gpu_usd = sum(
        float(item["maximum_usd"])
        for item in phase_maxima
        if str(item["command_phase"]) in planned_phase_set
    )
    if not all(
        math.isfinite(value)
        for value in (
            projected_compute_usd,
            projected_running_storage_usd,
            projected_gpu_usd,
        )
    ):
        raise PaidBundleError("projected GPU cost is non-finite")

    incurred = CostLedger.totals(ledger.document)
    committed = CostLedger.totals(ledger.document, include_estimates=True)
    if abs(committed["gpu"] - incurred["gpu"]) > 1e-6:
        raise PaidBundleError("fresh cumulative preview requires no outstanding GPU reservation")
    cumulative_gpu = round(committed["gpu"] + projected_gpu_usd, 6)
    safety_ceiling = float(
        (Decimal(str(bindings.caps_usd.gpu)) * (Decimal("1") - Decimal(str(gpu_safety_margin_fraction)))).quantize(
            Decimal("0.000001"),
            rounding=ROUND_FLOOR,
        )
    )
    cumulative_total = round(
        cumulative_gpu
        + bindings.caps_usd.api
        + committed["storage"]
        + committed["other"],
        6,
    )
    if cumulative_gpu > safety_ceiling + 1e-6:
        raise PaidBundleError(
            "current GPU spend plus all phase ceilings exceeds the safety-adjusted cap"
        )
    if cumulative_total > bindings.caps_usd.total + 1e-6:
        raise PaidBundleError("current commitments plus the approved plan exceed the total cap")
    cumulative = {
        "ledger_incurred": incurred,
        "ledger_committed": committed,
        "future_gpu_phase_maxima_usd": round(projected_gpu_usd, 6),
        "gpu_worst_case_usd": cumulative_gpu,
        "gpu_safety_margin_fraction": gpu_safety_margin_fraction,
        "gpu_safety_adjusted_ceiling_usd": safety_ceiling,
        "gpu_safety_headroom_usd": round(safety_ceiling - cumulative_gpu, 6),
        "gpu_hard_stop_headroom_usd": round(bindings.caps_usd.gpu - cumulative_gpu, 6),
        "api_hard_stop_usd": bindings.caps_usd.api,
        "total_worst_case_usd": cumulative_total,
        "total_hard_stop_headroom_usd": round(
            bindings.caps_usd.total - cumulative_total,
            6,
        ),
    }
    context_hashes = {
        "config": bindings.config_hash,
        "preregistration": bindings.preregistration_hash,
        "gpu_lock": bindings.gpu_lock_hash,
        "gpu_quote_lock": context.gpu_quote.content_hash,
        "api_quote_lock": context.api_quote.content_hash,
        "bindings": stable_hash(bindings.model_dump(mode="json")),
    }
    payload = PaidRunReviewPayload.model_validate(
        {
            "protocol_version": PAID_RUN_REVIEW_PROTOCOL_VERSION,
            "source_commit": source_commit_before,
            "context_hashes": context_hashes,
            "ledger": {
                "path": ledger.path.relative_to(context.project_root).as_posix(),
                "bytes_sha256": ledger.bytes_sha256,
                "document_hash": ledger.document_hash,
                "byte_count": len(ledger.raw_bytes),
            },
            "planned_command_phases": list(planned_command_phases),
            "phase_maxima_usd": list(phase_maxima),
            "caps_usd": bindings.caps_usd.model_dump(mode="json"),
            "cumulative_cost": cumulative,
        }
    )
    review = PaidRunReview(
        payload=payload,
        review_hash=paid_run_review_hash(payload),
    )
    source_commit_after = _clean_source_commit(
        context.project_root,
        ledger_path=ledger.path,
    )
    if source_commit_after != source_commit_before:
        raise PaidBundleError("project source commit changed during paid-run review")
    return ReviewComputation(
        review=review,
        planned_command_phases=planned_command_phases,
        phase_maxima=phase_maxima,
        projected_compute_usd=projected_compute_usd,
        projected_running_storage_usd=projected_running_storage_usd,
        projected_gpu_usd=projected_gpu_usd,
        cumulative=cumulative,
        gpu_quote_status=_timestamp_status(
            bindings.gpu.quote.quoted_at,
            maximum_age_seconds=MAX_GPU_QUOTE_AGE.total_seconds(),
        ),
        api_quote_status=_timestamp_status(
            bindings.api_quote.checked_at,
            maximum_age_seconds=MAX_API_QUOTE_AGE.total_seconds(),
        ),
    )


def _preview_payload(
    context: BundleContext,
    *,
    cost_ledger_path: Path,
    planned_command_phases: tuple[str, ...],
    gpu_safety_margin_fraction: float = 0.03,
) -> dict[str, Any]:
    computation = _review_computation(
        context,
        cost_ledger_path=cost_ledger_path,
        gpu_safety_margin_fraction=gpu_safety_margin_fraction,
        planned_command_phases=planned_command_phases,
    )
    bindings = context.bindings
    allocations = [item.model_dump(mode="json") for item in bindings.gpu.phase_runtime_allocations]
    return {
        "schema_version": 1,
        "approval_schema_version": APPROVAL_SCHEMA_VERSION,
        "status": "preview",
        "paid_execution_authorized": False,
        "ready_for_explicit_user_approval": computation.ready,
        "cumulative_cost_proven": True,
        "user_review_hash": computation.review.review_hash,
        "planned_command_phases": list(computation.planned_command_phases),
        "approval_review": computation.review.model_dump(mode="json"),
        "paths": {
            "config": _relative(context.config_path, context.project_root),
            "preregistration": _relative(
                context.preregistration_path,
                context.project_root,
            ),
            "gpu_lock": _relative(context.gpu_lock_path, context.project_root),
            "gpu_quote_lock": _relative(context.gpu_quote_path, context.project_root),
            "api_quote_lock": _relative(context.api_quote_path, context.project_root),
        },
        "hashes": {
            "config": bindings.config_hash,
            "preregistration": bindings.preregistration_hash,
            "gpu_lock": bindings.gpu_lock_hash,
            "gpu_quote_lock": context.gpu_quote.content_hash,
            "api_quote_lock": context.api_quote.content_hash,
            "bindings": stable_hash(bindings.model_dump(mode="json")),
        },
        "gpu": {
            "family": bindings.gpu.family,
            "provider_gpu_id": bindings.gpu.provider_gpu_id,
            "cloud_type": bindings.gpu.cloud_type,
            "allowed_cuda_versions": list(bindings.gpu.allowed_cuda_versions),
            "data_center_ids": list(bindings.gpu.data_center_ids),
            "count": bindings.gpu.count,
            "container_disk_gb": bindings.gpu.container_disk_gb,
            "volume_disk_gb": bindings.gpu.volume_disk_gb,
            "usd_per_gpu_hour": bindings.gpu.quote.usd_per_gpu_hour,
            "running_storage_usd_per_hour": (bindings.gpu.quote.running_storage_usd_per_hour),
            "phase_runtime_allocations": allocations,
            "phase_maxima_usd": list(computation.phase_maxima),
            "planned_phase_maxima_usd": [
                item
                for item in computation.phase_maxima
                if item["command_phase"] in computation.planned_command_phases
            ],
            "projected_compute_usd": round(computation.projected_compute_usd, 6),
            "projected_running_storage_usd": round(
                computation.projected_running_storage_usd,
                6,
            ),
            "projected_maximum_usd": round(computation.projected_gpu_usd, 6),
            "hard_stop_usd": bindings.caps_usd.gpu,
            "quote": computation.gpu_quote_status,
            "source_url": bindings.gpu.quote.source_url,
        },
        "api": {
            "routes": [route.model_dump(mode="json") for route in bindings.routes],
            "hard_stop_usd": bindings.caps_usd.api,
            "quote": computation.api_quote_status,
            "source_url": bindings.api_quote.source_url,
        },
        "total_hard_stop_usd": bindings.caps_usd.total,
        "cumulative_cost": computation.cumulative,
        "canonical_paid_phases": sorted(PAID_COMMAND_PHASES),
    }


def _approve(args: argparse.Namespace, context: BundleContext) -> dict[str, Any]:
    output = _private_path(
        Path(args.output),
        project_root=context.project_root,
        create_parent=True,
    )
    if output.name != APPROVAL_FILENAME:
        raise PaidBundleError(f"approval output must be named {APPROVAL_FILENAME}")
    if os.path.lexists(output):
        raise PaidBundleError(f"refusing to overwrite claimed approval: {output}")

    allowed_phases = canonicalize_paid_command_phases(tuple(args.allow_phase))
    supplied_review_hash = str(args.review_hash)
    if re.fullmatch(r"sha256:[0-9a-f]{64}", supplied_review_hash) is None:
        raise PaidBundleError("user-reviewed hash must be a namespaced SHA-256 hash")
    initial_review = _review_computation(
        context,
        cost_ledger_path=args.cost_ledger,
        gpu_safety_margin_fraction=args.gpu_safety_margin_fraction,
        planned_command_phases=allowed_phases,
    )
    if not initial_review.ready:
        raise PaidBundleError("both paid-provider quotes must be fresh at approval time")
    if not hmac.compare_digest(initial_review.review.review_hash, supplied_review_hash):
        raise PaidBundleError("current paid-run review does not match the user-reviewed hash")

    # Re-read every context input and the descriptor-stable ledger after the
    # first comparison. This closes the ordinary preview/approve TOCTOU window;
    # a concurrent source, quote, config, or ledger change cannot inherit an
    # approval for the earlier snapshot.
    confirmed_context = _load_context(args, allow_quote_specs=False)
    confirmed_review = _review_computation(
        confirmed_context,
        cost_ledger_path=args.cost_ledger,
        gpu_safety_margin_fraction=args.gpu_safety_margin_fraction,
        planned_command_phases=allowed_phases,
    )
    if (
        confirmed_context.bindings != context.bindings
        or confirmed_review.review != initial_review.review
        or not confirmed_review.ready
        or not hmac.compare_digest(confirmed_review.review.review_hash, supplied_review_hash)
    ):
        raise PaidBundleError("paid-run context changed while approval was being prepared")

    user_approval = UserApproval(
        approval_id=args.approval_id,
        approved_at=args.approved_at,
    )
    relevant_quote_times = []
    if any(phase.endswith("_gpu") for phase in allowed_phases):
        relevant_quote_times.append(confirmed_context.bindings.gpu.quote.quoted_at)
    if any(phase.endswith("_api") for phase in allowed_phases):
        relevant_quote_times.append(confirmed_context.bindings.api_quote.checked_at)
    if user_approval.approved_at < max(relevant_quote_times):
        raise PaidBundleError("user approval predates the reviewed provider quotes")
    provisional = {
        "schema_version": APPROVAL_SCHEMA_VERSION,
        "bindings": confirmed_context.bindings.model_dump(mode="json"),
        "review": confirmed_review.review.model_dump(mode="json"),
        "allowed_command_phases": list(allowed_phases),
        "user_approval": user_approval.model_dump(mode="json"),
    }
    provisional["content_hash"] = approval_content_hash(provisional)
    approval = PaidRunApproval.model_validate(provisional)
    now = datetime.now(UTC)
    for phase in allowed_phases:
        validate_paid_run_approval(
            approval,
            expected=confirmed_context.bindings,
            command_phase=phase,
            now=now,
            expected_source_commit=confirmed_review.review.payload.source_commit,
            expected_ledger_path=confirmed_review.review.payload.ledger.path,
        )

    final_review = _review_computation(
        confirmed_context,
        cost_ledger_path=args.cost_ledger,
        gpu_safety_margin_fraction=args.gpu_safety_margin_fraction,
        planned_command_phases=allowed_phases,
    )
    if final_review.review != confirmed_review.review or not final_review.ready:
        raise PaidBundleError("paid-run review changed before approval could be claimed")
    write_json_exclusive(output, approval.model_dump(mode="json"))
    loaded = load_paid_run_approval(output)
    if loaded != approval:
        raise PaidBundleError("claimed approval changed before it could be authenticated")
    for phase in allowed_phases:
        validate_paid_run_approval(
            loaded,
            expected=confirmed_context.bindings,
            command_phase=phase,
            now=now,
            expected_source_commit=confirmed_review.review.payload.source_commit,
            expected_ledger_path=confirmed_review.review.payload.ledger.path,
        )
    output.chmod(0o600)
    return {
        "schema_version": 1,
        "status": "approved",
        "approval_schema_version": APPROVAL_SCHEMA_VERSION,
        "approval_artifact_created": True,
        "paid_execution_authorized_for_listed_phases": True,
        "paid_execution_performed": False,
        "approval": _relative(output, context.project_root),
        "approval_content_hash": loaded.content_hash,
        "user_review_hash": loaded.review.review_hash,
        "source_commit": loaded.review.payload.source_commit,
        "ledger_bytes_sha256": loaded.review.payload.ledger.bytes_sha256,
        "ledger_document_hash": loaded.review.payload.ledger.document_hash,
        "approval_id_hash": stable_hash(loaded.user_approval.approval_id),
        "bindings_hash": stable_hash(loaded.bindings.model_dump(mode="json")),
        "allowed_command_phases": list(loaded.allowed_command_phases),
        "approved_at": loaded.user_approval.approved_at.astimezone(UTC)
        .isoformat()
        .replace("+00:00", "Z"),
        "secrets_in_output": False,
    }


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--gpu-lock", type=Path, required=True)
    parser.add_argument("--gpu-quote-lock", type=Path, required=True)
    parser.add_argument("--api-quote-lock", type=Path, required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)

    preview = subparsers.add_parser(
        "preview",
        help="freeze or authenticate quote locks and print a non-authorizing cost preview",
    )
    _add_common_arguments(preview)
    preview.add_argument(
        "--gpu-quote-spec",
        type=Path,
        help="private unhashed GPU quote JSON; requires --api-quote-spec",
    )
    preview.add_argument(
        "--api-quote-spec",
        type=Path,
        help="private unhashed API quote JSON; requires --gpu-quote-spec",
    )
    preview.add_argument(
        "--cost-ledger",
        type=Path,
        required=True,
        help="canonical ledger whose exact bytes bind the cumulative review",
    )
    preview.add_argument(
        "--gpu-safety-margin-fraction",
        type=float,
        default=0.03,
    )
    preview.add_argument(
        "--allow-phase",
        action="append",
        choices=sorted(PAID_COMMAND_PHASES),
        required=True,
        help="canonical phase included in this review window; repeat as needed",
    )

    approve = subparsers.add_parser(
        "approve",
        help="write one explicit user approval for already-previewed quote locks",
    )
    _add_common_arguments(approve)
    approve.add_argument("--output", type=Path, required=True)
    approve.add_argument("--cost-ledger", type=Path, required=True)
    approve.add_argument("--review-hash", required=True)
    approve.add_argument(
        "--gpu-safety-margin-fraction",
        type=float,
        default=0.03,
    )
    approve.add_argument("--approval-id", required=True)
    approve.add_argument("--approved-at", required=True)
    approve.add_argument(
        "--allow-phase",
        action="append",
        choices=sorted(PAID_COMMAND_PHASES),
        required=True,
        help="explicitly approved canonical phase; repeat for each phase",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    previous_umask = os.umask(0o077)
    try:
        config_path = _absolute(Path(args.config), base=Path.cwd())
        project_root = config_path.parent.parent
        (project_root / ".runpod").mkdir(mode=0o700, exist_ok=True)
        with paid_bundle_lock(project_root=project_root, exclusive=False):
            context = _load_context(args, allow_quote_specs=args.action == "preview")
            if args.action == "preview":
                result = _preview_payload(
                    context,
                    cost_ledger_path=args.cost_ledger,
                    planned_command_phases=canonicalize_paid_command_phases(
                        tuple(args.allow_phase)
                    ),
                    gpu_safety_margin_fraction=args.gpu_safety_margin_fraction,
                )
            else:
                result = _approve(args, context)
    except (
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        ValidationError,
        yaml.YAMLError,
    ) as exc:
        if isinstance(
            exc,
            (PaidBundleError, PaidRunApprovalError, PaidBundleRotationError, OSError),
        ):
            message = str(exc)
        else:
            message = "private paid-bundle validation failed; inspect the local input artifacts"
        parser.exit(2, f"error: {message}\n")
    finally:
        os.umask(previous_umask)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
