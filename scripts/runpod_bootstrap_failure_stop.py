#!/usr/bin/env python3
"""Stop a lifecycle-bound Pod when pre-bootstrap verification fails."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from model_forensics.runpod_bootstrap_failure import (  # noqa: E402
    BootstrapFailureStopError,
    stop_after_bootstrap_failure,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--reservation", type=Path, required=True)
    parser.add_argument("--expected-provider-gpu-id", required=True)
    parser.add_argument("--allowed-data-center-ids-csv", required=True)
    parser.add_argument("--expected-container-image", required=True)
    args = parser.parse_args()
    try:
        data_center_ids = tuple(args.allowed_data_center_ids_csv.split(","))
        summary = stop_after_bootstrap_failure(
            project_root=args.project_root,
            phase=args.phase,
            reservation_path=args.reservation,
            pod_id=os.environ.get("RUNPOD_POD_ID", ""),
            api_key=os.environ.get("RUNPOD_API_KEY", ""),
            session_nonce=os.environ.get("GPU_BUDGET_SESSION_ID", ""),
            expected_provider_gpu_id=args.expected_provider_gpu_id,
            allowed_data_center_ids=data_center_ids,
            expected_container_image=args.expected_container_image,
        )
    except (OSError, ValueError, BootstrapFailureStopError):
        print("bootstrap failure emergency stop was not confirmed", file=sys.stderr)
        return 2
    print(
        f"bootstrap failure emergency stop confirmed: {summary['pod_id_hash']}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
