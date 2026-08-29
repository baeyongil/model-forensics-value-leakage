#!/usr/bin/env python3
"""Freeze and validate the private quote/approval bundle without paid work.

``preview`` authenticates existing quote locks, or exclusively creates them
from two explicit unhashed JSON specs, then prints a secret-free cost summary.
``approve`` must be invoked separately after explicit user approval; it never
infers an approval identifier, timestamp, or command phase.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import stat
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
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
    ApprovalBindings,
    PaidRunApproval,
    PaidRunApprovalError,
    UserApproval,
    approval_content_hash,
    load_paid_run_approval,
    validate_paid_run_approval,
)
from model_forensics.config import RunConfig
from model_forensics.execution_bindings import (
    API_ROUTE_QUOTE_LOCK_FILENAME,
    GPU_QUOTE_LOCK_FILENAME,
    ApiRouteQuoteLock,
    GpuQuoteLock,
    api_route_quote_lock_content_hash,
    build_approval_bindings,
    gpu_quote_lock_content_hash,
    load_api_route_quote_lock,
    load_gpu_quote_lock,
)
from model_forensics.gpu_budget import write_json_exclusive
from model_forensics.io import stable_hash


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


def _load_yaml_mapping(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise PaidBundleError(f"{label} must be a regular non-symlink file: {path}")
    value = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueSafeLoader)
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


def _load_unhashed_quote_spec(path: Path, *, label: str) -> dict[str, Any]:
    _secure_existing_private_file(path, label=label)
    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_unique_json_object,
    )
    if not isinstance(value, dict):
        raise PaidBundleError(f"{label} must be a JSON object")
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
    gpu_spec_path: Path,
    api_spec_path: Path,
    gpu_quote_path: Path,
    api_quote_path: Path,
) -> None:
    gpu_raw = _load_unhashed_quote_spec(gpu_spec_path, label="GPU quote spec")
    api_raw = _load_unhashed_quote_spec(api_spec_path, label="API quote spec")
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
    config_path = Path(args.config).resolve()
    config_raw = _load_yaml_mapping(config_path, label="run config")
    config = RunConfig.model_validate(config_raw)
    config.source_path = config_path
    config.assert_execution_ready()
    project_root = config_path.parent.parent.resolve()

    preregistration_path = Path(args.preregistration).resolve()
    configured_preregistration = config.preregistration
    if not configured_preregistration.is_absolute():
        configured_preregistration = project_root / configured_preregistration
    if preregistration_path != configured_preregistration.resolve():
        raise PaidBundleError(
            "explicit preregistration path disagrees with the run config: "
            f"{preregistration_path} != {configured_preregistration.resolve()}"
        )
    preregistration = _load_yaml_mapping(
        preregistration_path,
        label="preregistration",
    )
    gpu_lock_path = Path(args.gpu_lock).resolve()
    gpu_lock = _load_yaml_mapping(gpu_lock_path, label="GPU/software lock")

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
            gpu_spec_path=gpu_spec_path,
            api_spec_path=api_spec_path,
            gpu_quote_path=gpu_quote_path,
            api_quote_path=api_quote_path,
        )

    _secure_existing_private_file(gpu_quote_path, label="GPU quote lock")
    _secure_existing_private_file(api_quote_path, label="API quote lock")
    gpu_quote = load_gpu_quote_lock(gpu_quote_path)
    api_quote = load_api_route_quote_lock(api_quote_path)
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


def _preview_payload(context: BundleContext) -> dict[str, Any]:
    bindings = context.bindings
    allocations = [item.model_dump(mode="json") for item in bindings.gpu.phase_runtime_allocations]
    allocated_hours = sum(float(item["maximum_runtime_hours"]) for item in allocations)
    projected_compute_usd = (
        bindings.gpu.count * bindings.gpu.quote.usd_per_gpu_hour * allocated_hours
    )
    projected_running_storage_usd = (
        bindings.gpu.quote.running_storage_usd_per_hour * allocated_hours
    )
    projected_gpu_usd = projected_compute_usd + projected_running_storage_usd
    if not all(
        math.isfinite(value)
        for value in (
            projected_compute_usd,
            projected_running_storage_usd,
            projected_gpu_usd,
        )
    ):
        raise PaidBundleError("projected GPU cost is non-finite")
    gpu_quote_status = _timestamp_status(
        bindings.gpu.quote.quoted_at,
        maximum_age_seconds=MAX_GPU_QUOTE_AGE.total_seconds(),
    )
    api_quote_status = _timestamp_status(
        bindings.api_quote.checked_at,
        maximum_age_seconds=MAX_API_QUOTE_AGE.total_seconds(),
    )
    return {
        "schema_version": 1,
        "approval_schema_version": APPROVAL_SCHEMA_VERSION,
        "status": "preview",
        "paid_execution_authorized": False,
        "ready_for_explicit_user_approval": bool(
            gpu_quote_status["fresh"] and api_quote_status["fresh"]
        ),
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
            "projected_compute_usd": round(projected_compute_usd, 6),
            "projected_running_storage_usd": round(
                projected_running_storage_usd,
                6,
            ),
            "projected_maximum_usd": round(projected_gpu_usd, 6),
            "hard_stop_usd": bindings.caps_usd.gpu,
            "quote": gpu_quote_status,
            "source_url": bindings.gpu.quote.source_url,
        },
        "api": {
            "routes": [route.model_dump(mode="json") for route in bindings.routes],
            "hard_stop_usd": bindings.caps_usd.api,
            "quote": api_quote_status,
            "source_url": bindings.api_quote.source_url,
        },
        "total_hard_stop_usd": bindings.caps_usd.total,
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

    allowed_phases = tuple(args.allow_phase)
    provisional = {
        "schema_version": APPROVAL_SCHEMA_VERSION,
        "bindings": context.bindings.model_dump(mode="json"),
        "allowed_command_phases": list(allowed_phases),
        "user_approval": UserApproval(
            approval_id=args.approval_id,
            approved_at=args.approved_at,
        ).model_dump(mode="json"),
    }
    provisional["content_hash"] = approval_content_hash(provisional)
    approval = PaidRunApproval.model_validate(provisional)
    now = datetime.now(UTC)
    for phase in allowed_phases:
        validate_paid_run_approval(
            approval,
            expected=context.bindings,
            command_phase=phase,
            now=now,
        )

    write_json_exclusive(output, approval.model_dump(mode="json"))
    loaded = load_paid_run_approval(output)
    for phase in allowed_phases:
        validate_paid_run_approval(
            loaded,
            expected=context.bindings,
            command_phase=phase,
            now=now,
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

    approve = subparsers.add_parser(
        "approve",
        help="write one explicit user approval for already-previewed quote locks",
    )
    _add_common_arguments(approve)
    approve.add_argument("--output", type=Path, required=True)
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
        context = _load_context(args, allow_quote_specs=args.action == "preview")
        if args.action == "preview":
            result = _preview_payload(context)
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
        if isinstance(exc, (PaidBundleError, PaidRunApprovalError, OSError)):
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
