#!/usr/bin/env python3
"""Create or validate the private reusable GPU software setup lock."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from model_forensics.gpu_setup import (
    GpuSetupSpec,
    create_gpu_setup_lock,
    validate_gpu_setup_lock,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("create", "validate"))
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--environment-manifest", type=Path, required=True)
    parser.add_argument("--venv-python", type=Path, required=True)
    parser.add_argument("--container-image-digest", required=True)
    parser.add_argument("--vllm-wheel-url", required=True)
    parser.add_argument("--vllm-wheel-sha256", required=True)
    parser.add_argument("--transformers-commit", required=True)
    parser.add_argument("--jlens-commit", required=True)
    args = parser.parse_args()

    private_root = (Path.cwd() / ".runpod").resolve()
    for path in (args.lock.resolve(), args.environment_manifest.resolve()):
        if not path.is_relative_to(private_root):
            raise SystemExit("GPU setup artifacts must remain under ignored .runpod/")
    try:
        spec = GpuSetupSpec(
            container_image_digest=args.container_image_digest,
            vllm_wheel_url=args.vllm_wheel_url,
            vllm_wheel_sha256=args.vllm_wheel_sha256,
            transformers_commit=args.transformers_commit,
            jlens_commit=args.jlens_commit,
        )
        operation = create_gpu_setup_lock if args.mode == "create" else validate_gpu_setup_lock
        if args.mode == "create":
            payload = operation(
                path=args.lock,
                spec=spec,
                environment_manifest_path=args.environment_manifest,
                venv_python_path=args.venv_python,
            )
        else:
            payload = operation(
                path=args.lock,
                expected_spec=spec,
                environment_manifest_path=args.environment_manifest,
                venv_python_path=args.venv_python,
            )
    except (OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
