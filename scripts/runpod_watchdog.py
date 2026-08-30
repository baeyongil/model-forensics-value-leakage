#!/usr/bin/env python3
"""Arm the independent RunPod GPU-cost watchdog."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from model_forensics.runpod_watchdog import (
    RunpodStopClient,
    WatchdogLimits,
    bind_lifecycle_pod,
    run_watchdog,
    wait_for_rearm_then_run_watchdog,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pod-id",
        help=(
            "optional ambient Pod id; when supplied it must equal the authenticated "
            "private lifecycle target"
        ),
    )
    parser.add_argument("--lifecycle-state", required=True)
    parser.add_argument("--expected-session-hash", required=True)
    parser.add_argument("--expected-phase", required=True)
    parser.add_argument(
        "--host-wait-for-rearm",
        action="store_true",
        help="acknowledge locally while EXITED, then watch a fresh re-arm/start",
    )
    parser.add_argument(
        "--host-rearm-ack",
        help="private acknowledgement written only after the stopped provider Pod is verified",
    )
    parser.add_argument("--running-readiness-timeout-seconds", type=float, default=300)
    parser.add_argument(
        "--expected-gpu-family",
        choices=("H100", "H100_80GB", "A100", "A100_80GB"),
        required=True,
    )
    parser.add_argument("--expected-provider-gpu-id", required=True)
    parser.add_argument("--allowed-data-center-id", action="append", required=True)
    parser.add_argument("--allowed-cuda-version", action="append", required=True)
    parser.add_argument("--expected-container-image", required=True)
    parser.add_argument("--expected-gpu-count", type=int, default=8)
    parser.add_argument("--maximum-approved-hourly-per-gpu-usd", required=True, type=float)
    parser.add_argument("--maximum-approved-storage-hourly-usd", required=True, type=float)
    parser.add_argument("--gpu-hard-stop-usd", required=True, type=float)
    parser.add_argument("--maximum-runtime-hours", required=True, type=float)
    parser.add_argument("--safety-margin-fraction", type=float, default=0.03)
    parser.add_argument("--prior-committed-gpu-usd", type=float, default=0.0)
    parser.add_argument("--poll-seconds", type=float, default=15)
    parser.add_argument("--state", required=True)
    parser.add_argument("--stop-request")
    parser.add_argument("--api-key-env", default="RUNPOD_API_KEY")
    parser.add_argument("--hf-token-env", default="HF_TOKEN")
    args = parser.parse_args(argv)
    bound_pod_id = bind_lifecycle_pod(
        lifecycle_state_path=args.lifecycle_state,
        expected_session_hash=args.expected_session_hash,
        expected_phase=args.expected_phase,
        ambient_pod_id=args.pod_id,
        waiting_for_rearm=args.host_wait_for_rearm,
    )
    approved_compute_hourly_usd = (
        args.maximum_approved_hourly_per_gpu_usd * args.expected_gpu_count
    )
    limits = WatchdogLimits(
        gpu_hard_stop_usd=args.gpu_hard_stop_usd,
        maximum_runtime_hours=args.maximum_runtime_hours,
        safety_margin_fraction=args.safety_margin_fraction,
        maximum_approved_hourly_total_usd=(
            approved_compute_hourly_usd + args.maximum_approved_storage_hourly_usd
        ),
        maximum_approved_compute_hourly_usd=approved_compute_hourly_usd,
        maximum_approved_storage_hourly_usd=args.maximum_approved_storage_hourly_usd,
        prior_committed_gpu_usd=args.prior_committed_gpu_usd,
    )
    client = RunpodStopClient(
        pod_id=bound_pod_id,
        expected_session_hash=args.expected_session_hash,
        api_key_env=args.api_key_env,
        hf_token_env=args.hf_token_env,
    )
    common = dict(
        pod_id=bound_pod_id,
        expected_gpu_family=args.expected_gpu_family,
        expected_provider_gpu_id=args.expected_provider_gpu_id,
        allowed_data_center_ids=tuple(args.allowed_data_center_id),
        allowed_cuda_versions=tuple(args.allowed_cuda_version),
        expected_container_image=args.expected_container_image,
        expected_gpu_count=args.expected_gpu_count,
        limits=limits,
        state_path=args.state,
        client=client,
        stop_request_path=args.stop_request,
        poll_seconds=args.poll_seconds,
    )
    if args.host_wait_for_rearm:
        if args.host_rearm_ack is None:
            parser.error("--host-wait-for-rearm requires --host-rearm-ack")
        wait_for_rearm_then_run_watchdog(
            lifecycle_state_path=args.lifecycle_state,
            expected_session_hash=args.expected_session_hash,
            expected_phase=args.expected_phase,
            acknowledgement_path=args.host_rearm_ack,
            running_readiness_timeout_seconds=args.running_readiness_timeout_seconds,
            **common,
        )
    elif args.host_rearm_ack is not None:
        parser.error("--host-rearm-ack requires --host-wait-for-rearm")
    else:
        run_watchdog(**common)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
