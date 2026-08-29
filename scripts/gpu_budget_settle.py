#!/usr/bin/env python3
"""Idempotently settle one stopped GPU session in the canonical cost ledger."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path

from model_forensics.budget import BudgetLimits, CostLedger
from model_forensics.gpu_budget import (
    GPU_PHASE_SETTLEMENT_PROTOCOL,
    GpuBudgetGateError,
    load_gpu_phase_budget_reservation,
    settle_gpu_phase_budget,
    validate_existing_gpu_phase_reservation,
    write_json_exclusive,
)
from model_forensics.io import read_json, stable_hash

_ENV_NAME_RE = re.compile(r"[A-Z][A-Z0-9_]{2,63}\Z")


def _private_runpod_path(path: Path, *, label: str) -> Path:
    private_root = (Path.cwd() / ".runpod").resolve()
    resolved = path.resolve()
    if not resolved.is_relative_to(private_root):
        raise ValueError(f"{label} must be under ignored .runpod/")
    return resolved


def _watchdog_stopped(path: Path) -> dict[str, object]:
    payload = read_json(path)
    if not isinstance(payload, dict) or payload.get("schema_version") != 2:
        raise GpuBudgetGateError("watchdog settlement state is malformed")
    if payload.get("watchdog_version") != "runpod-gpu-cost-watchdog-v2":
        raise GpuBudgetGateError("watchdog settlement state version is unsupported")
    if payload.get("status") != "stopped_confirmed":
        raise GpuBudgetGateError("GPU reservation cannot settle before stopped_confirmed")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reservation-receipt", type=Path, required=True)
    parser.add_argument("--cost-ledger", type=Path, required=True)
    parser.add_argument("--watchdog-state", type=Path, required=True)
    parser.add_argument("--session-id-env", default="GPU_BUDGET_SESSION_ID")
    parser.add_argument("--provider-incurred-usd", type=float, required=True)
    parser.add_argument("--gpu-hard-stop-usd", type=float, required=True)
    parser.add_argument("--api-hard-stop-usd", type=float, required=True)
    parser.add_argument("--total-hard-stop-usd", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    try:
        if _ENV_NAME_RE.fullmatch(args.session_id_env) is None:
            raise ValueError("GPU budget session environment variable name is invalid")
        session_id = os.environ.get(args.session_id_env)
        if not session_id:
            raise ValueError(
                "required opaque GPU budget session environment variable is unset: "
                f"{args.session_id_env}"
            )
        if not math.isfinite(args.provider_incurred_usd) or args.provider_incurred_usd < 0:
            raise ValueError("provider-incurred GPU cost must be finite and non-negative")
        receipt_path = _private_runpod_path(
            args.reservation_receipt,
            label="GPU reservation receipt",
        )
        watchdog_path = _private_runpod_path(
            args.watchdog_state,
            label="watchdog state",
        )
        output_path = _private_runpod_path(args.output, label="GPU settlement output")
        reservation = load_gpu_phase_budget_reservation(receipt_path)
        expected_session_directory = reservation.session_hash.removeprefix("sha256:")
        if watchdog_path.parent.name != expected_session_directory:
            raise GpuBudgetGateError("watchdog session directory disagrees with the reservation")
        watchdog = _watchdog_stopped(watchdog_path)
        ledger = CostLedger(
            args.cost_ledger,
            BudgetLimits(
                gpu=args.gpu_hard_stop_usd,
                api=args.api_hard_stop_usd,
                total=args.total_hard_stop_usd,
            ),
        )
        validate_existing_gpu_phase_reservation(
            ledger=ledger,
            reservation=reservation,
            phase=reservation.phase,
            session_id=session_id,
            require_active=False,
        )
        settle_gpu_phase_budget(
            ledger=ledger,
            reservation=reservation,
            incurred_usd=args.provider_incurred_usd,
        )
        payload = {
            "schema_version": 1,
            "protocol_version": GPU_PHASE_SETTLEMENT_PROTOCOL,
            "phase": reservation.phase,
            "reservation_id": reservation.reservation_id,
            "reservation_record_hash": reservation.manifest()["record_hash"],
            "session_hash": reservation.session_hash,
            "provider_incurred_usd": args.provider_incurred_usd,
            "watchdog_state_hash": stable_hash(watchdog),
            "status": "settled",
        }
        payload["record_hash"] = stable_hash(payload)
        if output_path.exists():
            observed = read_json(output_path)
            if observed != payload:
                raise GpuBudgetGateError(
                    "existing GPU settlement artifact disagrees with exact settlement"
                )
        else:
            write_json_exclusive(output_path, payload)
    except (OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
