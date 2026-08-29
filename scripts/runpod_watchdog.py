#!/usr/bin/env python3
"""Arm the independent RunPod GPU-cost watchdog."""

from __future__ import annotations

import argparse

from model_forensics.runpod_watchdog import (
    RunpodStopClient,
    WatchdogLimits,
    run_watchdog,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pod-id", required=True)
    parser.add_argument(
        "--expected-gpu-family",
        choices=("H100", "H100_80GB", "A100", "A100_80GB"),
        required=True,
    )
    parser.add_argument("--expected-provider-gpu-id", required=True)
    parser.add_argument("--allowed-data-center-id", action="append", required=True)
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
    args = parser.parse_args()
    limits = WatchdogLimits(
        gpu_hard_stop_usd=args.gpu_hard_stop_usd,
        maximum_runtime_hours=args.maximum_runtime_hours,
        safety_margin_fraction=args.safety_margin_fraction,
        maximum_approved_hourly_total_usd=(
            args.maximum_approved_hourly_per_gpu_usd * args.expected_gpu_count
            + args.maximum_approved_storage_hourly_usd
        ),
        maximum_approved_storage_hourly_usd=args.maximum_approved_storage_hourly_usd,
        prior_committed_gpu_usd=args.prior_committed_gpu_usd,
    )
    client = RunpodStopClient(pod_id=args.pod_id, api_key_env=args.api_key_env)
    run_watchdog(
        pod_id=args.pod_id,
        expected_gpu_family=args.expected_gpu_family,
        expected_provider_gpu_id=args.expected_provider_gpu_id,
        allowed_data_center_ids=tuple(args.allowed_data_center_id),
        expected_container_image=args.expected_container_image,
        expected_gpu_count=args.expected_gpu_count,
        limits=limits,
        state_path=args.state,
        client=client,
        stop_request_path=args.stop_request,
        poll_seconds=args.poll_seconds,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
