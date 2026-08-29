#!/usr/bin/env python3
"""Fail-closed RunPod hardware, price-freshness, and budget preflight."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import urllib.parse
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from model_forensics.budget import BudgetLimits, CostLedger
from model_forensics.gpu_budget import (
    load_gpu_phase_budget_reservation,
    validate_gpu_phase_bootstrap,
)
from model_forensics.runpod_watchdog import normalize_gpu_family

GPU_PATTERNS = {
    # Capacity is validated from memory.total. Some valid nvidia-smi names
    # (for example "NVIDIA H100 PCIe") do not repeat the 80 GB capacity.
    "H100_80GB": re.compile(r"\bH100\b", re.IGNORECASE),
    "A100_80GB": re.compile(r"\bA100\b", re.IGNORECASE),
}


def _command_output(command: list[str]) -> str:
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode:
        raise RuntimeError(completed.stderr.strip() or f"command failed: {command[0]}")
    return completed.stdout.strip()


def gpu_inventory() -> list[dict[str, float | int | str]]:
    output = _command_output(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,uuid,driver_version,mig.mode.current",
            "--format=csv,noheader,nounits",
        ]
    )
    inventory: list[dict[str, float | int | str]] = []
    for line in output.splitlines():
        index, name, memory_mib, uuid, driver, mig_mode = [
            part.strip() for part in line.split(",", 5)
        ]
        inventory.append(
            {
                "index": int(index),
                "name": name,
                "memory_gib": float(memory_mib) / 1024,
                "uuid": uuid,
                "driver_version": driver,
                "mig_mode": mig_mode,
            }
        )
    return inventory


def validate_inventory(
    inventory: list[dict[str, Any]],
    *,
    required_gpus: int,
    minimum_memory_gib: float,
    expected_gpu_family: str,
) -> None:
    if len(inventory) != required_gpus:
        raise ValueError(f"need exactly {required_gpus} visible GPUs, found {len(inventory)}")
    names = {str(gpu["name"]) for gpu in inventory}
    if len(names) != 1:
        raise ValueError(f"all GPUs must be homogeneous, observed {sorted(names)}")
    pattern = GPU_PATTERNS[expected_gpu_family]
    if not all(pattern.search(str(gpu["name"])) for gpu in inventory):
        raise ValueError(
            f"visible GPU does not match approved family {expected_gpu_family}: {sorted(names)}"
        )
    undersized = [gpu for gpu in inventory if float(gpu["memory_gib"]) < minimum_memory_gib]
    if undersized:
        raise ValueError(f"GPU memory below requirement: {undersized}")
    uuids = [str(gpu["uuid"]) for gpu in inventory]
    if len(set(uuids)) != required_gpus:
        raise ValueError("GPU UUIDs must be unique; MIG or duplicated inventory is unsupported")
    mig_enabled = [gpu for gpu in inventory if str(gpu.get("mig_mode", "")).lower() != "disabled"]
    if mig_enabled:
        raise ValueError("MIG must be disabled on every GPU for the 8-GPU profile")


def validate_cuda_visible_devices(value: str | None, *, required_gpus: int) -> None:
    if value is None:
        return
    devices = [item.strip() for item in value.split(",") if item.strip()]
    if len(devices) != required_gpus or len(set(devices)) != required_gpus:
        raise ValueError("CUDA_VISIBLE_DEVICES must expose exactly 8 unique devices when set")


def parse_fresh_price_timestamp(value: str, *, now: datetime | None = None) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("price checked timestamp must be ISO-8601 with timezone") from exc
    if parsed.tzinfo is None:
        raise ValueError("price checked timestamp must include timezone")
    current = now or datetime.now(UTC)
    age_seconds = (current - parsed.astimezone(UTC)).total_seconds()
    if age_seconds < -300:
        raise ValueError("price checked timestamp is in the future")
    if age_seconds > 6 * 3600:
        raise ValueError("RunPod price must have been checked within the last six hours")
    return parsed


def _parse_aware_timestamp(value: Any, *, field: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"watchdog {field} must be an ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"watchdog {field} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"watchdog {field} must include a timezone")
    return parsed.astimezone(UTC)


def _positive_finite(value: Any, *, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"watchdog {field} must be numeric")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"watchdog {field} must be numeric") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise ValueError(f"watchdog {field} must be positive and finite")
    return parsed


def validate_watchdog_state(
    payload: Any,
    *,
    expected_pod_id: str,
    expected_gpu_family: str,
    expected_gpu_count: int,
    planned_hours: float,
    approved_hourly_total_usd: float,
    gpu_budget_usd: float,
    expected_prior_committed_gpu_usd: float = 0.0,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Validate a fresh, live-metadata-backed state emitted by watchdog v2."""

    if not isinstance(payload, dict) or payload.get("schema_version") != 2:
        raise ValueError("watchdog state must use schema_version 2")
    if payload.get("watchdog_version") != "runpod-gpu-cost-watchdog-v2":
        raise ValueError("watchdog state version is not approved")
    if payload.get("pod_id") != expected_pod_id:
        raise ValueError("watchdog state targets a different Pod")
    if payload.get("status") != "armed":
        raise ValueError(f"watchdog must be armed; observed {payload.get('status')!r}")
    if payload.get("action") != "stop_only_preserve_volume":
        raise ValueError("watchdog action must be the non-destructive stop endpoint")
    current = (now or datetime.now(UTC)).astimezone(UTC)
    updated = _parse_aware_timestamp(payload.get("updated_at"), field="updated_at")
    age = (current - updated).total_seconds()
    if age < -300 or age > 90:
        raise ValueError("watchdog state must have been refreshed within 90 seconds")

    metadata = payload.get("live_metadata")
    if not isinstance(metadata, dict):
        raise ValueError("watchdog state is missing sanitized live metadata")
    if metadata.get("pod_id") != expected_pod_id:
        raise ValueError("watchdog live metadata targets a different Pod")
    if metadata.get("desired_status") != "RUNNING":
        raise ValueError("watchdog live metadata must report desiredStatus RUNNING")
    if metadata.get("locked") is not False:
        raise ValueError("watchdog live metadata must report an unlocked Pod")
    if metadata.get("gpu_count") != expected_gpu_count or expected_gpu_count != 8:
        raise ValueError("watchdog live metadata must report exactly 8 GPUs")
    family = normalize_gpu_family(expected_gpu_family)
    display_name = metadata.get("gpu_display_name")
    if (
        not isinstance(display_name, str)
        or re.search(
            rf"(?:^|[^A-Z0-9]){family}(?:$|[^A-Z0-9])",
            display_name,
            re.IGNORECASE,
        )
        is None
    ):
        raise ValueError("watchdog live GPU family does not match the local preflight profile")
    machine_identity = metadata.get("machine_gpu_identity")
    if (
        not isinstance(machine_identity, list)
        or not machine_identity
        or not all(isinstance(item, str) and item for item in machine_identity)
    ):
        raise ValueError("watchdog live metadata is missing machine GPU identity")
    recognized_machine_families = {
        candidate
        for item in machine_identity
        for candidate in ("H100", "A100")
        if re.search(rf"(?:^|[^A-Z0-9]){candidate}(?:$|[^A-Z0-9])", item, re.IGNORECASE)
    }
    if recognized_machine_families != {family}:
        raise ValueError("watchdog machine GPU identity disagrees with the approved family")
    live_nominal = _positive_finite(metadata.get("cost_per_hr"), field="cost_per_hr")
    live_effective = _positive_finite(
        metadata.get("adjusted_cost_per_hr"), field="adjusted_cost_per_hr"
    )
    if max(live_nominal, live_effective) > approved_hourly_total_usd + 0.01:
        raise ValueError("live RunPod hourly cost exceeds the approved quote")

    limits = payload.get("limits")
    deadline = payload.get("deadline")
    if not isinstance(limits, dict) or not isinstance(deadline, dict):
        raise ValueError("watchdog limits and deadline must be present")
    hard_stop = _positive_finite(limits.get("gpu_hard_stop_usd"), field="gpu_hard_stop_usd")
    safe_budget = _positive_finite(limits.get("safe_budget_usd"), field="safe_budget_usd")
    global_safe_budget = _positive_finite(
        limits.get("global_safe_budget_usd"), field="global_safe_budget_usd"
    )
    margin = _positive_finite(limits.get("safety_margin_fraction"), field="safety_margin_fraction")
    if margin >= 0.25:
        raise ValueError("watchdog safety margin must be below 0.25")
    raw_prior = limits.get("prior_committed_gpu_usd")
    if isinstance(raw_prior, bool):
        raise ValueError("watchdog prior_committed_gpu_usd must be numeric")
    try:
        prior_committed = float(raw_prior)
    except (TypeError, ValueError) as exc:
        raise ValueError("watchdog prior_committed_gpu_usd must be numeric") from exc
    if not math.isfinite(prior_committed) or prior_committed < 0:
        raise ValueError("watchdog prior_committed_gpu_usd must be finite and nonnegative")
    if abs(prior_committed - expected_prior_committed_gpu_usd) > 1e-6:
        raise ValueError("watchdog prior committed GPU cost disagrees with the canonical ledger")
    expected_global_safe = hard_stop * (1 - margin)
    expected_remaining_safe = expected_global_safe - prior_committed
    if (
        abs(hard_stop - gpu_budget_usd) > 1e-6
        or abs(global_safe_budget - expected_global_safe) > 1e-6
        or abs(safe_budget - expected_remaining_safe) > 1e-6
        or safe_budget <= 0
    ):
        raise ValueError("watchdog GPU budget does not match the approved hard stop")
    calculation_rate = _positive_finite(
        deadline.get("calculation_hourly_usd"), field="calculation_hourly_usd"
    )
    if calculation_rate + 1e-9 < live_effective:
        raise ValueError("watchdog calculation rate understates the live effective rate")
    effective_deadline = _parse_aware_timestamp(
        deadline.get("effective_deadline"), field="effective_deadline"
    )
    last_started_at = _parse_aware_timestamp(
        metadata.get("last_started_at"), field="live_metadata.last_started_at"
    )
    remaining_seconds = (effective_deadline - current).total_seconds()
    if remaining_seconds <= 0:
        raise ValueError("watchdog deadline has already elapsed")
    if last_started_at + timedelta(hours=planned_hours) > effective_deadline:
        raise ValueError("planned work does not fit inside the live watchdog deadline")
    incurred = deadline.get("incurred_cost_usd")
    if isinstance(incurred, bool) or not isinstance(incurred, (int, float)) or incurred < 0:
        raise ValueError("watchdog incurred cost must be nonnegative")
    projected_cost = calculation_rate * planned_hours
    if projected_cost > safe_budget:
        raise ValueError("live incurred cost plus planned work exceeds the safe GPU budget")
    return {
        "updated_at": updated.isoformat(),
        "live_nominal_hourly_usd": live_nominal,
        "live_effective_hourly_usd": live_effective,
        "calculation_hourly_usd": calculation_rate,
        "incurred_cost_usd": float(incurred),
        "effective_deadline": effective_deadline.isoformat(),
        "remaining_seconds": remaining_seconds,
        "projected_cost_usd": projected_cost,
        "safe_budget_usd": safe_budget,
        "global_safe_budget_usd": global_safe_budget,
        "prior_committed_gpu_usd": prior_committed,
    }


def validate_watchdog_pid(path: Path) -> int:
    try:
        raw = path.read_text(encoding="utf-8").strip()
        pid = int(raw)
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise ValueError("watchdog PID file is missing or invalid") from exc
    if pid <= 1:
        raise ValueError("watchdog PID must identify a non-system process")
    try:
        os.kill(pid, 0)
    except OSError as exc:
        raise ValueError("watchdog process is not alive") from exc
    return pid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--required-gpus", type=int, required=True)
    parser.add_argument("--minimum-memory-gib", type=float, required=True)
    parser.add_argument("--minimum-free-disk-gib", type=float, required=True)
    parser.add_argument("--expected-gpu-family", choices=sorted(GPU_PATTERNS), required=True)
    parser.add_argument("--pod-id", required=True)
    parser.add_argument("--watchdog-state", type=Path, required=True)
    parser.add_argument("--watchdog-pid-file", type=Path, required=True)
    parser.add_argument("--hourly-per-gpu-usd", type=float, required=True)
    parser.add_argument("--approved-phase-runtime-hours", type=float, required=True)
    parser.add_argument("--planned-hours", type=float, required=True)
    parser.add_argument("--gpu-budget-usd", type=float, required=True)
    parser.add_argument("--prior-committed-gpu-usd", type=float, default=0.0)
    parser.add_argument("--gpu-budget-reservation", type=Path, required=True)
    parser.add_argument("--cost-ledger", type=Path, required=True)
    parser.add_argument("--gpu-phase", required=True)
    parser.add_argument("--gpu-session-id-env", default="GPU_BUDGET_SESSION_ID")
    parser.add_argument("--api-budget-usd", type=float, required=True)
    parser.add_argument("--total-budget-usd", type=float, required=True)
    parser.add_argument("--price-source", required=True)
    parser.add_argument("--price-checked-at", required=True)
    parser.add_argument("--container-image-digest", required=True)
    parser.add_argument("--vllm-wheel-url", required=True)
    parser.add_argument("--vllm-wheel-sha256", required=True)
    parser.add_argument("--output", type=Path, default=Path("data/manifests/gpu_preflight.json"))
    args = parser.parse_args()

    if args.required_gpus != 8:
        raise SystemExit("primary/fallback profiles both require exactly 8 GPUs")
    for name, value in (
        ("minimum GPU memory", args.minimum_memory_gib),
        ("minimum free disk", args.minimum_free_disk_gib),
        ("hourly rate", args.hourly_per_gpu_usd),
        ("approved phase runtime", args.approved_phase_runtime_hours),
        ("planned hours", args.planned_hours),
        ("GPU budget", args.gpu_budget_usd),
        ("API budget", args.api_budget_usd),
        ("total budget", args.total_budget_usd),
    ):
        if not math.isfinite(value) or value <= 0:
            raise SystemExit(f"{name} must be positive and finite")
    if not re.fullmatch(r"[0-9a-f]{64}", args.vllm_wheel_sha256):
        raise SystemExit("vLLM wheel SHA-256 must be exactly 64 lowercase hex characters")
    if not re.fullmatch(r"[^\s]+@sha256:[0-9a-f]{64}", args.container_image_digest):
        raise SystemExit("container image must end in @sha256 plus 64 lowercase hex characters")
    for label, url in (("price source", args.price_source), ("vLLM wheel", args.vllm_wheel_url)):
        parsed_url = urllib.parse.urlparse(url)
        if (
            parsed_url.scheme != "https"
            or not parsed_url.netloc
            or parsed_url.username is not None
            or parsed_url.password is not None
            or parsed_url.fragment
        ):
            raise SystemExit(f"{label} must be a credential-free HTTPS URL without a fragment")
    wheel_url = urllib.parse.urlparse(args.vllm_wheel_url)
    if wheel_url.query:
        raise SystemExit("vLLM wheel URL must not contain query credentials or mutable parameters")
    if not wheel_url.path.endswith(".whl"):
        raise SystemExit("vLLM wheel URL path must end in .whl")
    checked_at = parse_fresh_price_timestamp(args.price_checked_at)

    try:
        if re.fullmatch(r"[A-Z][A-Z0-9_]{2,63}", args.gpu_session_id_env) is None:
            raise ValueError("GPU budget session environment variable name is invalid")
        session_id = os.environ.get(args.gpu_session_id_env)
        if not session_id:
            raise ValueError(
                "required opaque GPU budget session environment variable is unset: "
                f"{args.gpu_session_id_env}"
            )
        gpu_reservation = load_gpu_phase_budget_reservation(args.gpu_budget_reservation)
        gpu_budget_gate = validate_gpu_phase_bootstrap(
            ledger=CostLedger(
                args.cost_ledger,
                BudgetLimits(
                    gpu=args.gpu_budget_usd,
                    api=args.api_budget_usd,
                    total=args.total_budget_usd,
                ),
            ),
            reservation=gpu_reservation,
            phase=args.gpu_phase,
            session_id=session_id,
            expected_approved_runtime_hours=(args.approved_phase_runtime_hours),
            expected_live_hourly_total_usd=(args.required_gpus * args.hourly_per_gpu_usd),
        )
        if abs(args.planned_hours - gpu_reservation.maximum_safe_runtime_hours) > 1e-9:
            raise ValueError("planned hours disagree with the cumulative GPU reservation")
        if abs(args.prior_committed_gpu_usd - gpu_reservation.prior_committed_gpu_usd) > 1e-6:
            raise ValueError("prior committed GPU cost disagrees with the cumulative reservation")
        watchdog_payload = json.loads(args.watchdog_state.read_text(encoding="utf-8"))
        watchdog_pid = validate_watchdog_pid(args.watchdog_pid_file)
        watchdog = validate_watchdog_state(
            watchdog_payload,
            expected_pod_id=args.pod_id,
            expected_gpu_family=args.expected_gpu_family,
            expected_gpu_count=args.required_gpus,
            planned_hours=args.planned_hours,
            approved_hourly_total_usd=args.required_gpus * args.hourly_per_gpu_usd,
            gpu_budget_usd=args.gpu_budget_usd,
            expected_prior_committed_gpu_usd=args.prior_committed_gpu_usd,
        )
    except (OSError, RuntimeError, json.JSONDecodeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    inventory = gpu_inventory()
    try:
        validate_cuda_visible_devices(
            os.environ.get("CUDA_VISIBLE_DEVICES"), required_gpus=args.required_gpus
        )
        validate_inventory(
            inventory,
            required_gpus=args.required_gpus,
            minimum_memory_gib=args.minimum_memory_gib,
            expected_gpu_family=args.expected_gpu_family,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    free_disk_gib = shutil.disk_usage(Path.cwd()).free / 1024**3
    if free_disk_gib < args.minimum_free_disk_gib:
        raise SystemExit(
            f"need {args.minimum_free_disk_gib:.0f} GiB free disk, found {free_disk_gib:.1f} GiB"
        )
    quoted_cost = args.required_gpus * args.hourly_per_gpu_usd * args.planned_hours
    if quoted_cost > args.gpu_budget_usd:
        raise SystemExit(f"quoted GPU cost ${quoted_cost:.2f} exceeds ${args.gpu_budget_usd:.2f}")

    payload = {
        "schema_version": 3,
        "checked_at_utc": datetime.now(UTC).isoformat(),
        "pod_id": args.pod_id,
        "python": sys.version,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpus": inventory,
        "free_disk_gib": free_disk_gib,
        "price": {
            "approved_hourly_per_gpu_usd": args.hourly_per_gpu_usd,
            "approved_hourly_total_usd": args.required_gpus * args.hourly_per_gpu_usd,
            "live_nominal_hourly_total_usd": watchdog["live_nominal_hourly_usd"],
            "live_effective_hourly_total_usd": watchdog["live_effective_hourly_usd"],
            "source": args.price_source,
            "checked_at": checked_at.isoformat(),
        },
        "planned_hours": args.planned_hours,
        "quoted_gpu_cost_usd": round(quoted_cost, 6),
        "live_projected_gpu_cost_usd": round(watchdog["projected_cost_usd"], 6),
        "live_incurred_gpu_cost_usd": round(watchdog["incurred_cost_usd"], 6),
        "prior_committed_gpu_cost_usd": round(watchdog["prior_committed_gpu_usd"], 6),
        "gpu_budget_usd": args.gpu_budget_usd,
        "gpu_budget_reservation": gpu_budget_gate,
        "watchdog": {
            "pid": watchdog_pid,
            "state_path": str(args.watchdog_state),
            "state_updated_at": watchdog["updated_at"],
            "effective_deadline": watchdog["effective_deadline"],
            "remaining_seconds": round(watchdog["remaining_seconds"], 3),
            "safe_budget_usd": watchdog["safe_budget_usd"],
        },
        "container_image_digest": args.container_image_digest,
        "vllm_wheel": {
            "url": args.vllm_wheel_url,
            "sha256": args.vllm_wheel_sha256,
        },
        "passed": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
