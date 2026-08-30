#!/usr/bin/env python3
"""Idempotently settle one stopped GPU session in the canonical cost ledger."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from model_forensics.budget import BudgetLimits, CostLedger
from model_forensics.gpu_budget import (
    GpuBudgetGateError,
    load_gpu_phase_budget_reservation,
    settle_gpu_phase_budget,
    write_json_exclusive,
)
from model_forensics.io import read_json, stable_hash
from model_forensics.runpod_lifecycle_state import (
    authorization_from_state,
    load_lifecycle_state,
)
from model_forensics.runpod_no_start import load_no_start_receipt
from model_forensics.runpod_recovery import load_external_stop_receipt

GPU_PHASE_SETTLEMENT_V2_PROTOCOL = "cumulative-gpu-phase-settlement-v2"


def _private_runpod_path(path: Path, *, label: str) -> Path:
    private_root = (Path.cwd() / ".runpod").resolve()
    resolved = path.resolve()
    if not resolved.is_relative_to(private_root):
        raise ValueError(f"{label} must be under ignored .runpod/")
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reservation-receipt", type=Path, required=True)
    parser.add_argument("--cost-ledger", type=Path, required=True)
    parser.add_argument("--watchdog-state", type=Path)
    parser.add_argument("--external-stop-receipt", type=Path)
    parser.add_argument("--lifecycle-state", type=Path)
    parser.add_argument("--no-start-receipt", type=Path)
    # Accepted only to produce a deliberate fail-closed migration error. No
    # legacy watchdog state or caller amount is ever parsed or used below.
    parser.add_argument("--provider-incurred-usd", type=float)
    parser.add_argument("--gpu-hard-stop-usd", type=float, required=True)
    parser.add_argument("--api-hard-stop-usd", type=float, required=True)
    parser.add_argument("--total-hard-stop-usd", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    try:
        legacy = args.watchdog_state is not None
        external = args.external_stop_receipt is not None
        no_start = args.no_start_receipt is not None
        if legacy or args.provider_incurred_usd is not None:
            raise ValueError(
                "legacy watchdog/provider-amount settlement is disabled; use an "
                "authenticated external-stop receipt"
            )
        if external == no_start:
            raise ValueError("provide exactly one authenticated external-stop or no-start receipt")
        if args.lifecycle_state is None:
            raise ValueError(
                "settlement requires the authenticated stopped lifecycle"
            )
        receipt_path = _private_runpod_path(
            args.reservation_receipt,
            label="GPU reservation receipt",
        )
        output_path = _private_runpod_path(args.output, label="GPU settlement output")
        reservation = load_gpu_phase_budget_reservation(receipt_path)
        expected_session_directory = reservation.session_hash.removeprefix("sha256:")
        expected_reservation_path = (
            Path.cwd() / ".runpod" / "reservations" / f"{reservation.phase}.json"
        ).resolve()
        expected_output_path = (
            Path.cwd() / ".runpod" / "sessions" / expected_session_directory / "settlement.json"
        ).resolve()
        expected_ledger_path = (Path.cwd() / "data" / "manifests" / "cost_ledger.yaml").resolve()
        if receipt_path != expected_reservation_path:
            raise GpuBudgetGateError("GPU reservation receipt path is not canonical for its phase")
        if Path(args.cost_ledger).resolve() != expected_ledger_path:
            raise GpuBudgetGateError("GPU cost ledger path is not canonical")
        if output_path != expected_output_path:
            raise GpuBudgetGateError("settlement output path is not canonical for the reservation")
        if external:
            assert args.external_stop_receipt is not None
            assert args.lifecycle_state is not None
            external_path = _private_runpod_path(
                args.external_stop_receipt,
                label="external-stop receipt",
            )
            if external_path.parent.name != expected_session_directory:
                raise GpuBudgetGateError(
                    "external-stop session directory disagrees with the reservation"
                )
            expected_external_path = (
                Path.cwd()
                / ".runpod"
                / "sessions"
                / expected_session_directory
                / "external_stop_receipt.json"
            ).resolve()
            if external_path != expected_external_path:
                raise GpuBudgetGateError(
                    "external-stop receipt path is not canonical for the reservation"
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
            lifecycle_path = _private_runpod_path(
                args.lifecycle_state,
                label="stopped lifecycle state",
            )
            expected_lifecycle_path = (Path.cwd() / ".runpod" / "pod_lifecycle.json").resolve()
            if lifecycle_path != expected_lifecycle_path:
                raise GpuBudgetGateError("stopped lifecycle path is not canonical")
            lifecycle = load_lifecycle_state(lifecycle_path)
            lifecycle_authorization = authorization_from_state(lifecycle)
            pod = lifecycle.get("pod")
            if (
                lifecycle.get("operation") != "stopped"
                or not isinstance(pod, dict)
                or pod.get("status") != "EXITED"
                or lifecycle.get("record_hash") != external_receipt.get("lifecycle_stopped_hash")
                or lifecycle_authorization.phase != reservation.phase
                or lifecycle_authorization.session_hash != reservation.session_hash
                or lifecycle_authorization.reservation_id != reservation.reservation_id
                or lifecycle_authorization.reservation_record_hash
                != reservation.manifest()["record_hash"]
            ):
                raise GpuBudgetGateError(
                    "external-stop receipt and stopped lifecycle bindings disagree"
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
            no_start_receipt = None
        else:
            assert args.no_start_receipt is not None
            no_start_path = _private_runpod_path(
                args.no_start_receipt,
                label="no-start receipt",
            )
            if no_start_path.parent.name != expected_session_directory:
                raise GpuBudgetGateError(
                    "no-start session directory disagrees with the reservation"
                )
            no_start_receipt = load_no_start_receipt(no_start_path)
            for field, expected in (
                ("session_hash", reservation.session_hash),
                ("reservation_id", reservation.reservation_id),
                ("reservation_record_hash", reservation.manifest()["record_hash"]),
            ):
                if no_start_receipt.get(field) != expected:
                    raise GpuBudgetGateError(
                        f"no-start receipt {field} disagrees with the reservation"
                    )
            if no_start_receipt.get("accounted_gpu_usd") != 0.0:
                raise GpuBudgetGateError("no-start receipt must account exactly zero GPU cost")
            assert args.lifecycle_state is not None
            lifecycle_path = _private_runpod_path(
                args.lifecycle_state,
                label="stopped lifecycle state",
            )
            expected_lifecycle_path = (Path.cwd() / ".runpod" / "pod_lifecycle.json").resolve()
            if lifecycle_path != expected_lifecycle_path:
                raise GpuBudgetGateError("stopped lifecycle path is not canonical")
            lifecycle = load_lifecycle_state(lifecycle_path)
            lifecycle_authorization = authorization_from_state(lifecycle)
            pod = lifecycle.get("pod")
            if (
                lifecycle.get("operation") != "stopped"
                or not isinstance(pod, dict)
                or pod.get("status") != "EXITED"
                or lifecycle.get("record_hash")
                != no_start_receipt.get("lifecycle_stopped_hash")
                or lifecycle_authorization.phase != reservation.phase
                or lifecycle_authorization.session_hash != reservation.session_hash
                or lifecycle_authorization.reservation_id != reservation.reservation_id
                or lifecycle_authorization.reservation_record_hash
                != reservation.manifest()["record_hash"]
            ):
                raise GpuBudgetGateError(
                    "no-start receipt and stopped lifecycle bindings disagree"
                )
            incurred_usd = 0.0
            external_receipt = None
        ledger = CostLedger(
            args.cost_ledger,
            BudgetLimits(
                gpu=args.gpu_hard_stop_usd,
                api=args.api_hard_stop_usd,
                total=args.total_hard_stop_usd,
            ),
        )
        if external:
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
        else:
            assert no_start_receipt is not None
            payload = {
                "schema_version": 2,
                "protocol_version": GPU_PHASE_SETTLEMENT_V2_PROTOCOL,
                "phase": reservation.phase,
                "reservation_id": reservation.reservation_id,
                "reservation_record_hash": reservation.manifest()["record_hash"],
                "session_hash": reservation.session_hash,
                "provider_incurred_usd": 0.0,
                "accounted_gpu_usd": 0.0,
                "billing_status": "not_started",
                "evidence_kind": "provider_no_start",
                "no_start_receipt_hash": no_start_receipt["record_hash"],
                "provider_evidence_hash": no_start_receipt["provider_evidence_hash"],
                "billing_evidence_hash": no_start_receipt["billing_evidence_hash"],
                "status": "settled",
            }
        payload["record_hash"] = stable_hash(payload)
        if output_path.exists():
            observed = read_json(output_path)
            if observed != payload:
                raise GpuBudgetGateError("existing GPU settlement artifact has different content")
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
