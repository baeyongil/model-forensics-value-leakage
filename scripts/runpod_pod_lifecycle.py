#!/usr/bin/env python3
"""Create, inspect, or re-arm the one approved RunPod Pod.

All paid mutations are guarded by the independently constructed approval
bindings and a pre-created active GPU budget reservation.  Provider and model
tokens are read from the environment and never printed or persisted.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from model_forensics.approval import (
    APPROVAL_FILENAME,
    ApprovalBindings,
    PaidRunApproval,
    load_paid_run_approval,
)
from model_forensics.budget import BudgetLimits, CostLedger
from model_forensics.config import RunConfig
from model_forensics.execution_bindings import (
    API_ROUTE_QUOTE_LOCK_FILENAME,
    GPU_QUOTE_LOCK_FILENAME,
    build_approval_bindings,
    load_api_route_quote_lock,
    load_gpu_quote_lock,
)
from model_forensics.gpu_budget import (
    GpuPhaseBudgetReservation,
    load_gpu_phase_budget_reservation,
)
from model_forensics.runpod_lifecycle import (
    HttpTransport,
    RunpodLifecycleClient,
    authorize_gpu_lifecycle,
    create_approved_pod,
    lifecycle_state_path,
    read_lifecycle_status,
    rearm_approved_pod,
    urllib_http_transport,
)

_ENV_NAME_RE = re.compile(r"[A-Z][A-Z0-9_]{2,63}\Z")


class LifecycleCliError(ValueError):
    """Local launch inputs are incomplete, unsafe, or inconsistent."""


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
            raise LifecycleCliError("YAML mapping contains an unhashable key") from exc
        if duplicate:
            raise LifecycleCliError("YAML mapping contains a duplicate key")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueSafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _load_yaml_mapping(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise LifecycleCliError(f"{label} must be a regular non-symlink file")
    value = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueSafeLoader)
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise LifecycleCliError(f"{label} must be a string-keyed YAML mapping")
    return value


def _absolute(raw: str | Path, *, root: Path) -> Path:
    path = Path(raw)
    return Path(os.path.abspath(os.fspath(path if path.is_absolute() else root / path)))


def _secure_private_input(raw: str | Path, *, root: Path, label: str) -> Path:
    private_root = lifecycle_state_path(root).parent
    path = _absolute(raw, root=root)
    try:
        path.relative_to(private_root)
    except ValueError as exc:
        raise LifecycleCliError(f"{label} must remain under the private .runpod directory") from exc
    current = private_root
    for component in path.relative_to(private_root).parts:
        current = current / component
        if os.path.lexists(current) and current.is_symlink():
            raise LifecycleCliError(f"{label} path must not contain a symlink")
    if not path.is_file():
        raise LifecycleCliError(f"{label} is missing")
    details = path.lstat()
    if not stat.S_ISREG(details.st_mode) or details.st_nlink != 1:
        raise LifecycleCliError(f"{label} must be a regular, non-hard-linked file")
    if details.st_uid != os.getuid():
        raise LifecycleCliError(f"{label} has an unexpected owner")
    path.chmod(0o600)
    return path


@dataclass(frozen=True, slots=True)
class PaidContext:
    root: Path
    bindings: ApprovalBindings
    approval: PaidRunApproval
    reservation: GpuPhaseBudgetReservation
    ledger: CostLedger


def _load_paid_context(args: argparse.Namespace) -> PaidContext:
    root = Path(args.project_root).resolve()
    if not root.is_dir():
        raise LifecycleCliError("project root is not a directory")
    config_path = _absolute(args.config, root=root)
    config = RunConfig.model_validate(_load_yaml_mapping(config_path, label="run config"))
    config.source_path = config_path
    config.assert_execution_ready()

    preregistration_path = _absolute(args.preregistration, root=root)
    configured_preregistration = config.preregistration
    if not configured_preregistration.is_absolute():
        configured_preregistration = root / configured_preregistration
    if preregistration_path != configured_preregistration.resolve():
        raise LifecycleCliError("preregistration path disagrees with the run config")
    preregistration = _load_yaml_mapping(preregistration_path, label="preregistration")
    gpu_lock = _load_yaml_mapping(
        _absolute(args.gpu_lock, root=root),
        label="GPU/software lock",
    )

    gpu_quote_path = _secure_private_input(
        args.gpu_quote_lock,
        root=root,
        label="GPU quote lock",
    )
    api_quote_path = _secure_private_input(
        args.api_quote_lock,
        root=root,
        label="API quote lock",
    )
    approval_path = _secure_private_input(
        args.approval,
        root=root,
        label="paid-run approval",
    )
    reservation_path = _secure_private_input(
        args.reservation,
        root=root,
        label="GPU reservation receipt",
    )
    if gpu_quote_path.name != GPU_QUOTE_LOCK_FILENAME:
        raise LifecycleCliError(f"GPU quote lock must be named {GPU_QUOTE_LOCK_FILENAME}")
    if api_quote_path.name != API_ROUTE_QUOTE_LOCK_FILENAME:
        raise LifecycleCliError(
            f"API quote lock must be named {API_ROUTE_QUOTE_LOCK_FILENAME}"
        )
    if approval_path.name != APPROVAL_FILENAME:
        raise LifecycleCliError(f"approval must be named {APPROVAL_FILENAME}")

    gpu_quote = load_gpu_quote_lock(gpu_quote_path)
    api_quote = load_api_route_quote_lock(api_quote_path)
    bindings = build_approval_bindings(
        config=config,
        preregistration=preregistration,
        gpu_lock=gpu_lock,
        quote_lock=gpu_quote,
        api_quote_lock=api_quote,
    )
    approval = load_paid_run_approval(approval_path)
    reservation = load_gpu_phase_budget_reservation(reservation_path)
    ledger = CostLedger(
        _absolute(args.cost_ledger, root=root),
        BudgetLimits(
            gpu=float(config.execution.gpu_cost_hard_stop_usd),
            api=float(config.execution.api_cost_hard_stop_usd),
            total=float(config.execution.total_cost_hard_stop_usd),
        ),
    )
    return PaidContext(
        root=root,
        bindings=bindings,
        approval=approval,
        reservation=reservation,
        ledger=ledger,
    )


def _secret_from_environment(name: str, *, label: str) -> str:
    if _ENV_NAME_RE.fullmatch(name) is None:
        raise LifecycleCliError(f"{label} environment variable name is invalid")
    value = os.environ.get(name)
    if not value:
        raise LifecycleCliError(f"required {label} environment variable is unset")
    return value


def _add_paid_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default="config/run_122b.yaml")
    parser.add_argument("--preregistration", default="config/preregistration.yaml")
    parser.add_argument("--gpu-lock", default="config/gpu_lock.yaml")
    parser.add_argument("--gpu-quote-lock", default=f".runpod/{GPU_QUOTE_LOCK_FILENAME}")
    parser.add_argument(
        "--api-quote-lock",
        default=f".runpod/{API_ROUTE_QUOTE_LOCK_FILENAME}",
    )
    parser.add_argument("--approval", default=f".runpod/{APPROVAL_FILENAME}")
    parser.add_argument("--reservation", required=True)
    parser.add_argument("--cost-ledger", default="cost_ledger.yaml")
    parser.add_argument("--phase", required=True)
    parser.add_argument("--session-id-env", default="GPU_BUDGET_SESSION_ID")
    parser.add_argument("--hf-token-env", default="HF_TOKEN")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fail-closed lifecycle for the one approved RunPod Pod"
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--api-key-env", default="RUNPOD_API_KEY")
    subparsers = parser.add_subparsers(dest="operation", required=True)

    create = subparsers.add_parser("create", help="create the one exact approved Pod")
    _add_paid_arguments(create)
    create.add_argument("--name")
    create.add_argument(
        "--allow-existing-pod-id-hash",
        action="append",
        default=[],
        help=(
            "repeat once per user-confirmed unrelated nonterminal Pod; value must be "
            "runpod-pod-id-sha256:<64 lowercase hex>"
        ),
    )

    subparsers.add_parser("status", help="read and validate status without mutation")

    rearm = subparsers.add_parser("rearm", help="re-arm and start the same stopped Pod")
    _add_paid_arguments(rearm)
    return parser


def _safe_output(value: Mapping[str, Any], *, root: Path) -> dict[str, Any]:
    state = lifecycle_state_path(root)
    return {**dict(value), "private_state": str(state.relative_to(root))}


def main(
    argv: Sequence[str] | None = None,
    *,
    transport: HttpTransport = urllib_http_transport,
) -> int:
    args = _parser().parse_args(argv)
    try:
        root = Path(args.project_root).resolve()
        api_key = _secret_from_environment(args.api_key_env, label="RunPod API key")
        if args.operation == "status":
            client = RunpodLifecycleClient(api_key=api_key, transport=transport)
            result = read_lifecycle_status(project_root=root, client=client)
        else:
            context = _load_paid_context(args)
            nonce = _secret_from_environment(args.session_id_env, label="GPU session nonce")
            hf_token = _secret_from_environment(args.hf_token_env, label="Hugging Face token")
            authorization = authorize_gpu_lifecycle(
                approval=context.approval,
                expected_bindings=context.bindings,
                reservation=context.reservation,
                ledger=context.ledger,
                phase=args.phase,
                session_nonce=nonce,
            )
            # Construct the provider client only after every offline approval,
            # quote, immutable-spec, ledger, reservation, and nonce gate passes.
            client = RunpodLifecycleClient(api_key=api_key, transport=transport)
            if args.operation == "create":
                name = args.name or f"model-forensics-{args.phase.removesuffix('_gpu')}"
                result = create_approved_pod(
                    project_root=context.root,
                    client=client,
                    authorization=authorization,
                    name=name,
                    hf_token=hf_token,
                    session_nonce=nonce,
                    acknowledged_existing_pod_id_hashes=(
                        args.allow_existing_pod_id_hash
                    ),
                )
            else:
                result = rearm_approved_pod(
                    project_root=context.root,
                    client=client,
                    authorization=authorization,
                    ledger=context.ledger,
                    hf_token=hf_token,
                    session_nonce=nonce,
                )
        print(json.dumps(_safe_output(result, root=root), ensure_ascii=True, sort_keys=True))
        return 0
    except (OSError, RuntimeError, ValueError) as exc:
        # All underlying errors are deliberately secret-free; provider error
        # bodies and request bodies are never interpolated.
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    raise SystemExit(main())
