#!/usr/bin/env python3
"""Extract authenticated GPU bootstrap values using only Python's stdlib.

The official RunPod image is not assumed to contain Pydantic or PyYAML before
the project environment is installed.  This narrow reader validates the
content-addressed GPU quote and the three immutable YAML scalars needed to arm
the watchdog.  Full approval validation still runs through the project models.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

GPU_PHASES = (
    "behavior_baseline_gpu",
    "behavior_treatment_gpu",
    "resample_gpu",
    "lens_gpu",
)
QUOTE_KEYS = {
    "schema_version",
    "provider",
    "quote_id",
    "gpu_family",
    "provider_gpu_id",
    "cloud_type",
    "allowed_cuda_versions",
    "data_center_ids",
    "gpu_count",
    "container_disk_gb",
    "volume_disk_gb",
    "usd_per_gpu_hour",
    "running_storage_usd_per_hour",
    "quoted_at",
    "phase_runtime_allocations",
    "source_url",
    "content_hash",
}
_RAW_HASH_RE = re.compile(r"[0-9a-f]{64}\Z")
_IMAGE_RE = re.compile(r"[^\s@]+@sha256:[0-9a-f]{64}\Z")
_DATA_CENTER_RE = re.compile(r"[A-Z0-9][A-Z0-9-]{2,31}\Z")
_QUERY_COMPONENT_RE = re.compile(r"[A-Za-z0-9._~-]+\Z")


class BootstrapInputError(ValueError):
    """The frozen launch inputs cannot safely arm a Pod."""


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise BootstrapInputError(f"duplicate JSON key in GPU quote lock: {key}")
        result[key] = value
    return result


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _aware_timestamp(value: Any) -> datetime:
    if not isinstance(value, str):
        raise BootstrapInputError("quoted_at must be an ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise BootstrapInputError("quoted_at must be ISO-8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise BootstrapInputError("quoted_at must include a timezone")
    return parsed.astimezone(UTC)


def _positive_float(value: Any, *, field: str) -> float:
    if isinstance(value, bool):
        raise BootstrapInputError(f"{field} must be numeric")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise BootstrapInputError(f"{field} must be numeric") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise BootstrapInputError(f"{field} must be positive and finite")
    return parsed


def _credential_free_https(
    value: Any,
    *,
    field: str,
    allow_deterministic_query: bool = False,
) -> str:
    if not isinstance(value, str) or any(
        ord(character) <= 0x20 or ord(character) == 0x7F for character in value
    ):
        raise BootstrapInputError(f"{field} must be a single-line URL")
    parsed = urlsplit(value)
    if (
        parsed.scheme != "https"
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
    ):
        raise BootstrapInputError(f"{field} must be a credential-free immutable HTTPS URL")
    if parsed.query:
        if not allow_deterministic_query:
            raise BootstrapInputError(
                f"{field} must be a credential-free immutable HTTPS URL"
            )
        pairs = parsed.query.split("&")
        if any(pair.count("=") != 1 for pair in pairs):
            raise BootstrapInputError(f"{field} query must be deterministic and credential-free")
        components = [pair.split("=", 1) for pair in pairs]
        if any(
            _QUERY_COMPONENT_RE.fullmatch(key) is None
            or _QUERY_COMPONENT_RE.fullmatch(component_value) is None
            for key, component_value in components
        ) or len({key for key, _ in components}) != len(components):
            raise BootstrapInputError(f"{field} query must be deterministic and credential-free")
    return value


def load_quote(path: Path, *, phase: str) -> dict[str, Any]:
    if path.name != "gpu_quote_lock.json":
        raise BootstrapInputError("GPU quote lock must be named gpu_quote_lock.json")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    except (OSError, json.JSONDecodeError) as exc:
        raise BootstrapInputError("cannot read GPU quote lock") from exc
    if not isinstance(payload, dict) or set(payload) != QUOTE_KEYS:
        raise BootstrapInputError("GPU quote lock has an unexpected schema")
    unsigned = {key: value for key, value in payload.items() if key != "content_hash"}
    quoted_at = _aware_timestamp(unsigned.get("quoted_at"))
    unsigned["quoted_at"] = quoted_at.isoformat().replace("+00:00", "Z")
    if payload.get("content_hash") != _stable_hash(unsigned):
        raise BootstrapInputError("GPU quote lock content hash mismatch")
    if payload.get("schema_version") != 1 or payload.get("provider") != "runpod":
        raise BootstrapInputError("GPU quote lock provider/schema is unsupported")
    if payload.get("gpu_family") not in {"H100_80GB", "A100_80GB"}:
        raise BootstrapInputError("GPU quote lock family is unsupported")
    provider_gpu_id = payload.get("provider_gpu_id")
    if (
        not isinstance(provider_gpu_id, str)
        or not provider_gpu_id.strip()
        or any(character in provider_gpu_id for character in "\t\r\n")
    ):
        raise BootstrapInputError("provider_gpu_id is invalid")
    if payload.get("cloud_type") != "SECURE":
        raise BootstrapInputError("GPU quote lock must require Secure Cloud")
    if payload.get("allowed_cuda_versions") != ["12.8"]:
        raise BootstrapInputError("GPU quote lock must allow exactly CUDA 12.8")
    data_centers = payload.get("data_center_ids")
    if (
        not isinstance(data_centers, list)
        or not data_centers
        or len(set(data_centers)) != len(data_centers)
        or any(not isinstance(item, str) or _DATA_CENTER_RE.fullmatch(item) is None for item in data_centers)
    ):
        raise BootstrapInputError("GPU quote data_center_ids are invalid")
    if payload.get("gpu_count") != 8:
        raise BootstrapInputError("GPU quote lock must require exactly eight GPUs")
    if payload.get("container_disk_gb") != 50 or payload.get("volume_disk_gb") != 650:
        raise BootstrapInputError("GPU quote lock storage sizes must be exactly 50/650 GB")
    compute_rate = _positive_float(payload.get("usd_per_gpu_hour"), field="usd_per_gpu_hour")
    storage_rate = _positive_float(
        payload.get("running_storage_usd_per_hour"),
        field="running_storage_usd_per_hour",
    )
    allocations = payload.get("phase_runtime_allocations")
    if not isinstance(allocations, list) or tuple(
        item.get("command_phase") if isinstance(item, dict) else None for item in allocations
    ) != GPU_PHASES:
        raise BootstrapInputError("GPU quote phase allocations are incomplete or reordered")
    selected = next(item for item in allocations if item["command_phase"] == phase)
    if set(selected) != {"command_phase", "maximum_runtime_hours"}:
        raise BootstrapInputError("GPU quote phase allocation has an unexpected schema")
    runtime = _positive_float(
        selected.get("maximum_runtime_hours"),
        field="maximum_runtime_hours",
    )
    source_url = _credential_free_https(
        payload.get("source_url"),
        field="source_url",
        allow_deterministic_query=True,
    )
    return {
        "gpu_family": payload["gpu_family"],
        "provider_gpu_id": provider_gpu_id,
        "allowed_cuda_versions_csv": "12.8",
        "data_center_ids_csv": ",".join(data_centers),
        "running_storage_usd_per_hour": format(storage_rate, ".17g"),
        "usd_per_gpu_hour": format(compute_rate, ".17g"),
        "maximum_runtime_hours": format(runtime, ".17g"),
        "source_url": source_url,
        "quoted_at": quoted_at.isoformat(),
    }


def _one_match(pattern: str, text: str, *, field: str) -> str:
    matches = re.findall(pattern, text, flags=re.MULTILINE)
    if len(matches) != 1:
        raise BootstrapInputError(f"GPU lock must contain exactly one {field}")
    return str(matches[0]).strip()


def load_gpu_lock(path: Path) -> dict[str, str]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise BootstrapInputError("cannot read GPU/software lock") from exc
    if "\t" in text:
        raise BootstrapInputError("GPU/software lock must not contain tabs")
    container = _one_match(
        r"^  reference:\s*([^\s#]+)\s*$",
        text,
        field="container image reference",
    )
    wheel_hash = _one_match(
        r"^    wheel_sha256:\s*([0-9a-f]+)\s*$",
        text,
        field="vLLM wheel SHA-256",
    )
    wheel_url_marker = re.search(r"^    wheel_url:\s*>-\s*$", text, flags=re.MULTILINE)
    if wheel_url_marker is None:
        wheel_url = _one_match(
            r"^    wheel_url:\s*([^\s#]+)\s*$",
            text,
            field="vLLM wheel URL",
        )
    else:
        tail = text[wheel_url_marker.end() :].splitlines()
        continuation: list[str] = []
        for line in tail:
            if not line.strip():
                continue
            indentation = len(line) - len(line.lstrip(" "))
            if indentation <= 4:
                break
            continuation.append(line.strip())
        if not continuation:
            raise BootstrapInputError("folded vLLM wheel URL is empty")
        wheel_url = " ".join(continuation)
    if _IMAGE_RE.fullmatch(container) is None:
        raise BootstrapInputError("GPU lock container image is not pinned by digest")
    if _RAW_HASH_RE.fullmatch(wheel_hash) is None:
        raise BootstrapInputError("GPU lock vLLM wheel SHA-256 is invalid")
    wheel_url = _credential_free_https(wheel_url, field="vLLM wheel URL")
    if not wheel_url.endswith(".whl"):
        raise BootstrapInputError("GPU lock vLLM URL must identify an exact wheel")
    return {
        "container_image_digest": container,
        "vllm_wheel_url": wheel_url,
        "vllm_wheel_sha256": wheel_hash,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-quote-lock", type=Path, required=True)
    parser.add_argument("--gpu-lock", type=Path, required=True)
    parser.add_argument("--phase", choices=GPU_PHASES, required=True)
    args = parser.parse_args()
    try:
        values = {
            **load_quote(args.gpu_quote_lock, phase=args.phase),
            **load_gpu_lock(args.gpu_lock),
        }
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    ordered = (
        "gpu_family",
        "provider_gpu_id",
        "allowed_cuda_versions_csv",
        "data_center_ids_csv",
        "running_storage_usd_per_hour",
        "usd_per_gpu_hour",
        "maximum_runtime_hours",
        "source_url",
        "quoted_at",
        "container_image_digest",
        "vllm_wheel_url",
        "vllm_wheel_sha256",
    )
    print("\t".join(values[key] for key in ordered))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
