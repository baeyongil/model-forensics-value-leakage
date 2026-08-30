#!/usr/bin/env python3
"""Archive an expired private paid bundle without provider access."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from model_forensics.paid_bundle_rotation import (
    PaidBundleRotationError,
    rotate_paid_bundle,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Provider-free, fail-closed rotation of canonical paid quote locks, optional "
            "approval, and optional quote specs."
        )
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--bundle-id",
        default=os.environ.get("PAID_BUNDLE_ID") or None,
        help=(
            "Explicit safe archive ID. If omitted, a content-derived sha256-* ID is used; "
            "an incomplete content-derived transaction is discovered and resumed."
        ),
    )
    args = parser.parse_args()
    try:
        result = rotate_paid_bundle(
            project_root=args.project_root,
            bundle_id=args.bundle_id,
        )
    except (OSError, ValueError, PaidBundleRotationError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
