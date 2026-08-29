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
SEMANTIC_DISTRIBUTIONS = (
    "huggingface-hub",
    "numpy",
    "scikit-learn",
    "scipy",
    "sentence-transformers",
    "tokenizers",
    "torch",
    "transformers",
)
SEMANTIC_DISTRIBUTION_VERSION = "5.7.0"
SEMANTIC_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
SEMANTIC_MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
SEMANTIC_PROTOCOL_VERSION = "pinned-sentence-transformer-runtime-v2"
SEMANTIC_STACK_VERSIONS = {
    "huggingface-hub": "1.29.0",
    "numpy": "2.5.2",
    "scikit-learn": "1.9.0",
    "scipy": "1.18.1",
    "sentence-transformers": SEMANTIC_DISTRIBUTION_VERSION,
    "tokenizers": "0.23.1",
    "torch": "2.13.0",
    "transformers": "5.16.0.dev0",
}
SEMANTIC_WHEEL_SHA256 = "b78141da3d8137e70d965866e2ca43190b9266f3d4d8752e250ded75e7136730"
SEMANTIC_WHEEL_URL = (
    "https://files.pythonhosted.org/packages/e8/c8/"
    "f63d99e354532f5b83e735dd1e001bda92495fbfde934f65d924abf2b071/"
    "sentence_transformers-5.7.0-py3-none-any.whl"
)
BOOTSTRAP_PROTOCOL_VERSION = "exact-direct-bootstrap-v1"
BOOTSTRAP_CONSTRAINTS_PATH = "config/gpu_bootstrap_constraints.txt"
BOOTSTRAP_CONSTRAINTS_SHA256 = (
    "eac31466ffe0668c030ef18bb2e88444ad2ff264733ae2526787db30148db662"
)
BOOTSTRAP_DISTRIBUTION_VERSIONS = {
    "accelerate": "1.12.0",
    "huggingface-hub": "1.29.0",
    "jlens": "0.1.0",
    "matplotlib": "3.10.8",
    "numpy": "2.5.2",
    "pandas": "3.0.3",
    "pydantic": "2.12.5",
    "PyYAML": "6.0.3",
    "safetensors": "0.7.0",
    "scikit-learn": "1.9.0",
    "scipy": "1.18.1",
    "sentence-transformers": "5.7.0",
    "setuptools": "80.9.0",
    "tokenizers": "0.23.1",
    "torch": "2.13.0",
    "transformers": "5.16.0.dev0",
    "vllm": "0.28.0",
    "wheel": "0.46.3",
}


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


def _folded_or_scalar(text: str, *, key: str, field: str) -> str:
    marker = re.search(rf"^  {re.escape(key)}:\s*>-\s*$", text, flags=re.MULTILINE)
    if marker is None:
        return _one_match(
            rf"^  {re.escape(key)}:\s*([^\s#]+)\s*$",
            text,
            field=field,
        )
    tail = text[marker.end() :].splitlines()
    continuation: list[str] = []
    for line in tail:
        if not line.strip():
            continue
        indentation = len(line) - len(line.lstrip(" "))
        if indentation <= 2:
            break
        continuation.append(line.strip())
    if not continuation:
        raise BootstrapInputError(f"folded {field} is empty")
    return " ".join(continuation)


def load_gpu_lock(path: Path) -> dict[str, str]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise BootstrapInputError("cannot read GPU/software lock") from exc
    if "\t" in text:
        raise BootstrapInputError("GPU/software lock must not contain tabs")
    if _one_match(r"^schema_version:\s*([0-9]+)\s*$", text, field="schema version") != "3":
        raise BootstrapInputError("GPU/software lock schema is unsupported")
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
    semantic_url = _folded_or_scalar(
        text,
        key="semantic_wheel_url",
        field="semantic wheel URL",
    )
    semantic_hash = _one_match(
        r"^  semantic_wheel_sha256:\s*([0-9a-f]+)\s*$",
        text,
        field="semantic wheel SHA-256",
    )
    semantic_version = _one_match(
        r'^  semantic_distribution_version:\s*"([0-9A-Za-z.+-]+)"\s*$',
        text,
        field="semantic distribution version",
    )
    semantic_protocol = _one_match(
        r"^  protocol_version:\s*([^\s#]+)\s*$",
        text,
        field="semantic runtime protocol",
    )
    semantic_model_id = _one_match(
        r"^  semantic_model_id:\s*([^\s#]+)\s*$",
        text,
        field="semantic model ID",
    )
    semantic_model_revision = _one_match(
        r"^  semantic_model_revision:\s*([0-9a-f]+)\s*$",
        text,
        field="semantic model revision",
    )
    stack_hash = _one_match(
        r"^  stack_lock_hash:\s*(sha256:[0-9a-f]+)\s*$",
        text,
        field="semantic stack lock hash",
    )
    section = re.search(
        r"^  required_distribution_versions:\s*$\n(?P<body>(?:^    [^\n]+\n)+)^  stack_lock_hash:",
        text,
        flags=re.MULTILINE,
    )
    if section is None:
        raise BootstrapInputError("GPU lock semantic distribution version set is absent")
    versions: dict[str, str] = {}
    for line in section.group("body").splitlines():
        match = re.fullmatch(r'    ([a-z0-9-]+):\s*"([0-9A-Za-z.+-]+)"', line)
        if match is None or match.group(1) in versions:
            raise BootstrapInputError("GPU lock semantic distribution version row is malformed")
        versions[match.group(1)] = match.group(2)
    if tuple(sorted(versions)) != SEMANTIC_DISTRIBUTIONS:
        raise BootstrapInputError("GPU lock semantic distribution inventory drifted")
    if stack_hash != _stable_hash(dict(sorted(versions.items()))):
        raise BootstrapInputError("GPU lock semantic stack hash mismatch")
    if versions != SEMANTIC_STACK_VERSIONS:
        raise BootstrapInputError("GPU lock semantic distribution versions drifted")
    semantic_url = _credential_free_https(semantic_url, field="semantic wheel URL")
    if not semantic_url.endswith(".whl") or _RAW_HASH_RE.fullmatch(semantic_hash) is None:
        raise BootstrapInputError("GPU lock semantic wheel artifact is invalid")
    if (
        semantic_protocol != SEMANTIC_PROTOCOL_VERSION
        or semantic_url != SEMANTIC_WHEEL_URL
        or semantic_hash != SEMANTIC_WHEEL_SHA256
        or semantic_version != SEMANTIC_DISTRIBUTION_VERSION
        or semantic_model_id != SEMANTIC_MODEL_ID
        or semantic_model_revision != SEMANTIC_MODEL_REVISION
    ):
        raise BootstrapInputError("GPU lock semantic runtime disagrees with the compiled lock")
    bootstrap_protocol = _one_match(
        r"^  bootstrap_protocol_version:\s*([^\s#]+)\s*$",
        text,
        field="bootstrap protocol",
    )
    constraints_path = _one_match(
        r"^  constraints_path:\s*([^\s#]+)\s*$", text, field="bootstrap constraints path"
    )
    constraints_hash = _one_match(
        r"^  constraints_sha256:\s*([0-9a-f]+)\s*$",
        text,
        field="bootstrap constraints SHA-256",
    )
    bootstrap_lock_hash = _one_match(
        r"^  distribution_lock_hash:\s*(sha256:[0-9a-f]+)\s*$",
        text,
        field="bootstrap distribution lock hash",
    )
    bootstrap_section = re.search(
        r"^bootstrap_environment:\s*$\n(?P<body>(?:^  .*\n|^    .*\n)+?)^container_image:",
        text,
        flags=re.MULTILINE,
    )
    if bootstrap_section is None:
        raise BootstrapInputError("GPU bootstrap environment lock is absent")
    bootstrap_versions_section = re.search(
        r"^  required_distribution_versions:\s*$\n(?P<body>(?:^    [^\n]+\n)+)^  distribution_lock_hash:",
        bootstrap_section.group("body"),
        flags=re.MULTILINE,
    )
    if bootstrap_versions_section is None:
        raise BootstrapInputError("GPU bootstrap distribution inventory is absent")
    bootstrap_versions: dict[str, str] = {}
    for line in bootstrap_versions_section.group("body").splitlines():
        match = re.fullmatch(r'    ([A-Za-z0-9-]+):\s*"([0-9A-Za-z.+-]+)"', line)
        if match is None or match.group(1) in bootstrap_versions:
            raise BootstrapInputError("GPU bootstrap distribution row is malformed")
        bootstrap_versions[match.group(1)] = match.group(2)
    if (
        bootstrap_protocol != BOOTSTRAP_PROTOCOL_VERSION
        or constraints_path != BOOTSTRAP_CONSTRAINTS_PATH
        or constraints_hash != BOOTSTRAP_CONSTRAINTS_SHA256
        or bootstrap_versions != BOOTSTRAP_DISTRIBUTION_VERSIONS
        or bootstrap_lock_hash != _stable_hash(dict(sorted(bootstrap_versions.items())))
    ):
        raise BootstrapInputError("GPU bootstrap environment disagrees with the compiled lock")
    return {
        "container_image_digest": container,
        "vllm_wheel_url": wheel_url,
        "vllm_wheel_sha256": wheel_hash,
        "semantic_wheel_url": semantic_url,
        "semantic_wheel_sha256": semantic_hash,
        "semantic_distribution_version": semantic_version,
        "semantic_stack_lock_hash": stack_hash,
        "bootstrap_constraints_path": constraints_path,
        "bootstrap_constraints_sha256": constraints_hash,
        "bootstrap_distribution_lock_hash": bootstrap_lock_hash,
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
        "semantic_wheel_url",
        "semantic_wheel_sha256",
        "semantic_distribution_version",
        "semantic_stack_lock_hash",
        "bootstrap_constraints_path",
        "bootstrap_constraints_sha256",
        "bootstrap_distribution_lock_hash",
    )
    print("\t".join(values[key] for key in ordered))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
