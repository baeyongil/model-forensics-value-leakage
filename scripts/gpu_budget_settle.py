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
from model_forensics.runpod_recovery import load_external_stop_receipt

_ENV_NAME_RE = re.compile(r"[A-Z][A-Z0-9_]{2,63}\Z")
GPU_PHASE_SETTLEMENT_V2_PROTOCOL = "cumulative-gpu-phase-settlement-v2"


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
    parser.add_argument("--watchdog-state", type=Path)
    parser.add_argument("--external-stop-receipt", type=Path)
    parser.add_argument("--session-id-env", default="GPU_BUDGET_SESSION_ID")
    parser.add_argument("--provider-incurred-usd", type=float)
    parser.add_argument("--gpu-hard-stop-usd", type=float, required=True)
    parser.add_argument("--api-hard-stop-usd", type=float, required=True)
    parser.add_argument("--total-hard-stop-usd", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    try:
        if _ENV_NAME_RE.fullmatch(args.session_id_env) is None:
            raise ValueError("GPU budget session environment variable name is invalid")
        legacy = args.watchdog_state is not None
        external = args.external_stop_receipt is not None
        if legacy == external:
            raise ValueError(
                "provide exactly one of --watchdog-state or --external-stop-receipt"
            )
        if legacy and args.provider_incurred_usd is None:
            raise ValueError("legacy watchdog settlement requires --provider-incurred-usd")
        if external and args.provider_incurred_usd is not None:
            raise ValueError(
                "external-stop settlement derives its amount from authenticated evidence"
            )
        session_id = os.environ.get(args.session_id_env)
        if legacy and not session_id:
            raise ValueError(
                "required opaque GPU budget session environment variable is unset: "
                f"{args.session_id_env}"
            )
        if legacy and (
            not math.isfinite(args.provider_incurred_usd)
            or args.provider_incurred_usd < 0
        ):
            raise ValueError("provider-incurred GPU cost must be finite and non-negative")
        receipt_path = _private_runpod_path(
            args.reservation_receipt,
            label="GPU reservation receipt",
        )
        output_path = _private_runpod_path(args.output, label="GPU settlement output")
        reservation = load_gpu_phase_budget_reservation(receipt_path)
        expected_session_directory = reservation.session_hash.removeprefix("sha256:")
        if output_path.parent.name != expected_session_directory:
            raise GpuBudgetGateError(
                "settlement output session directory disagrees with the reservation"
            )
        if legacy:
            assert args.watchdog_state is not None
            watchdog_path = _private_runpod_path(
                args.watchdog_state,
                label="watchdog state",
            )
            if watchdog_path.parent.name != expected_session_directory:
                raise GpuBudgetGateError(
                    "watchdog session directory disagrees with the reservation"
                )
            watchdog = _watchdog_stopped(watchdog_path)
            incurred_usd = float(args.provider_incurred_usd)
            external_receipt = None
        else:
            assert args.external_stop_receipt is not None
            external_path = _private_runpod_path(
                args.external_stop_receipt,
                label="external-stop receipt",
            )
            if external_path.parent.name != expected_session_directory:
                raise GpuBudgetGateError(
                    "external-stop session directory disagrees with the reservation"
                )
            external_receipt = load_external_stop_receipt(external_path)
            for field, expected in (
                ("session_hash", reservation.session_hash),
                ("reservation_id", reservation.reservation_id),
                ("reservation_record_hash", reservation.manifest()["record_hash"]),
            ):
                if external_receipt.get(field) != expected:
                    raise GpuBudgetGateError(
                        f"external-stop receipt {field} disagrees with the reservation"
                    )
            incurred_raw = external_receipt.get("settlement_amount_usd")
            if (
                isinstance(incurred_raw, bool)
                or not isinstance(incurred_raw, (int, float))
                or not math.isfinite(float(incurred_raw))
                or float(incurred_raw) < 0
            ):
                raise GpuBudgetGateError("external-stop settlement amount is invalid")
            incurred_usd = float(incurred_raw)
            watchdog = None
        ledger = CostLedger(
            args.cost_ledger,
            BudgetLimits(
                gpu=args.gpu_hard_stop_usd,
                api=args.api_hard_stop_usd,
                total=args.total_hard_stop_usd,
            ),
        )
        if legacy:
            assert session_id is not None
            validate_existing_gpu_phase_reservation(
                ledger=ledger,
                reservation=reservation,
                phase=reservation.phase,
                session_id=session_id,
                require_active=False,
            )
        if legacy:
            assert watchdog is not None
            payload = {
                "schema_version": 1,
                "protocol_version": GPU_PHASE_SETTLEMENT_PROTOCOL,
                "phase": reservation.phase,
                "reservation_id": reservation.reservation_id,
                "reservation_record_hash": reservation.manifest()["record_hash"],
                "session_hash": reservation.session_hash,
                "provider_incurred_usd": incurred_usd,
                "watchdog_state_hash": stable_hash(watchdog),
                "status": "settled",
            }
        else:
            assert external_receipt is not None
            billing = external_receipt.get("billing_evidence")
            provider_incurred = (
                billing.get("provider_amount_usd") if isinstance(billing, dict) else None
            )
            payload = {
                "schema_version": 2,
                "protocol_version": GPU_PHASE_SETTLEMENT_V2_PROTOCOL,
                "phase": reservation.phase,
                "reservation_id": reservation.reservation_id,
                "reservation_record_hash": reservation.manifest()["record_hash"],
                "session_hash": reservation.session_hash,
                "provider_incurred_usd": provider_incurred,
                "accounted_gpu_usd": incurred_usd,
                "billing_status": external_receipt["billing_status"],
                "evidence_kind": external_receipt["evidence_kind"],
                "external_stop_receipt_hash": external_receipt["record_hash"],
                "stop_evidence_hash": external_receipt["stop_evidence_hash"],
                "billing_evidence_hash": external_receipt["billing_evidence_hash"],
                "status": "settled",
            }
        payload["record_hash"] = stable_hash(payload)
        if output_path.exists():
            observed = read_json(output_path)
            if observed != payload:
                raise GpuBudgetGateError(
                    "existing GPU settlement artifact has different content"
                )
        settle_gpu_phase_budget(
            ledger=ledger,
            reservation=reservation,
            incurred_usd=incurred_usd,
        )
        if not output_path.exists():
            write_json_exclusive(output_path, payload)
    except (OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
