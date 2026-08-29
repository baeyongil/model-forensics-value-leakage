#!/usr/bin/env python3
"""Fail-closed audit of files proposed for the public repository.

The default candidate set is the union of tracked files and untracked files that
are not ignored by Git.  This deliberately catches a secret or raw artifact
*before* the first commit as well as files accidentally force-added later.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path, PurePosixPath
from typing import Any

import yaml
from yaml.constructor import ConstructorError
from yaml.resolver import BaseResolver

SCHEMA_VERSION = 1
DEFAULT_MAX_FILE_BYTES = 10 * 1024 * 1024

REQUIRED_RELEASE_PATHS = frozenset(
    {
        ".gitignore",
        "LICENSE",
        "README.md",
        "THIRD_PARTY_NOTICES.md",
        "config/preregistration.yaml",
    }
)

ROOT_ALLOWLIST = frozenset(
    {
        ".env.example",
        ".gitignore",
        "CITATION.cff",
        "LICENSE",
        "Makefile",
        "README.md",
        "RUNPOD.md",
        "THIRD_PARTY_NOTICES.md",
        "pyproject.toml",
    }
)

ALLOWED_TREES: tuple[tuple[str, frozenset[str]], ...] = (
    ("config/", frozenset({".yaml", ".yml", ".json", ".txt"})),
    ("docs/", frozenset({".md"})),
    ("scripts/", frozenset({".py", ".sh"})),
    ("src/model_forensics/", frozenset({".py", ".typed"})),
    ("tests/", frozenset({".py"})),
    ("reports/figures/", frozenset({".png", ".svg", ".pdf"})),
)

# Every public manifest is reviewed by exact path.  In particular, this does
# not include RunPod watchdog/preflight/environment state, paid-response
# checkpoints, or any raw/interim artifact.  Adding a newly generated manifest
# to the public repository therefore requires an explicit code review here;
# merely giving it a JSON/YAML suffix is never sufficient.
PUBLIC_MANIFEST_ALLOWLIST = frozenset(
    {
        "data/manifests/adjudication_manifest.jsonl",
        "data/manifests/anchor_classifications_locked.json",
        "data/manifests/anchor_manifest.json",
        "data/manifests/anchor_prefilter_manifest.json",
        "data/manifests/behavioral_final_consensus.jsonl",
        "data/manifests/behavioral_final_consensus_summary.json",
        "data/manifests/behavioral_quality_gate.json",
        "data/manifests/cost_ledger.yaml",
        "data/manifests/first_estimate_spans.jsonl",
        "data/manifests/independent_final_manifest.jsonl",
        "data/manifests/lens_compatibility_manifest.json",
        "data/manifests/lens_execution_manifest.json",
        "data/manifests/lens_position_manifest.json",
        "data/manifests/lens_positions.jsonl",
        "data/manifests/lens_probe_token_verification.json",
        "data/manifests/lens_validation.json",
        "data/manifests/qwen4b_prefix_gpu_smoke.json",
        "data/manifests/resampling_execution_lock.json",
        "data/manifests/resampling_execution_manifest.json",
        "data/manifests/resampling_initial_allocation.json",
        "data/manifests/resampling_stage_two_allocation.json",
        "data/manifests/resampling_validation.json",
        "data/manifests/sampling_execution_lock.json",
        "data/manifests/sampling_manifest.json",
        "data/manifests/smoke/adjudication_manifest.jsonl",
        "data/manifests/smoke/anchor_manifest.json",
        "data/manifests/smoke/sampling_manifest.json",
        "data/manifests/smoke/smoke_completion.json",
        "data/manifests/threshold_manifests.json",
        "data/manifests/time_ledger.yaml",
        "data/manifests/upstream_reference.json",
        "data/manifests/upstream_reference_summary.json",
    }
)

FORBIDDEN_PREFIXES = (
    ".git/",
    ".omx/",
    ".pytest_cache/",
    ".ruff_cache/",
    ".runpod/",
    ".venv/",
    ".venv-gpu/",
    "data/cache/",
    "data/interim/",
    "data/raw/",
    "data/upstream/",
    "outputs/",
    "reports/rendered/",
    "reports/staging/",
    "secrets/",
)

FORBIDDEN_SUFFIXES = frozenset(
    {
        ".bin",
        ".ckpt",
        ".dill",
        ".gguf",
        ".key",
        ".onnx",
        ".p12",
        ".pem",
        ".pickle",
        ".pkl",
        ".pt",
        ".pth",
        ".safetensors",
        ".tar",
        ".tgz",
        ".whl",
        ".zip",
    }
)

TEXT_SUFFIXES = frozenset(
    {
        "",
        ".cff",
        ".json",
        ".jsonl",
        ".md",
        ".py",
        ".sh",
        ".svg",
        ".toml",
        ".txt",
        ".yaml",
        ".yml",
    }
)

TOKEN_PATTERNS: tuple[tuple[str, re.Pattern[bytes]], ...] = (
    ("private_key", re.compile(rb"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("openrouter_token", re.compile(rb"sk-or-v1-[A-Za-z0-9_-]{20,}")),
    ("openai_token", re.compile(rb"sk-(?:proj-)?[A-Za-z0-9_-]{20,}")),
    ("hugging_face_token", re.compile(rb"hf_[A-Za-z0-9]{20,}")),
    ("github_fine_grained_token", re.compile(rb"github_pat_[A-Za-z0-9_]{20,}")),
    ("github_classic_token", re.compile(rb"gh[pousr]_[A-Za-z0-9]{20,}")),
    ("aws_access_key", re.compile(rb"(?:AKIA|ASIA)[A-Z0-9]{16}")),
    ("slack_token", re.compile(rb"xox[baprs]-[A-Za-z0-9-]{20,}")),
)

ENV_ASSIGNMENT = re.compile(
    r"^[ \t]*(?:export[ \t]+)?[A-Z][A-Z0-9_]*(?:KEY|TOKEN|SECRET|PASSWORD)"
    r"[ \t]*=[ \t]*([^\r\n]*?)[ \t]*$",
    re.MULTILINE,
)
GENERIC_ASSIGNMENT = re.compile(
    r"(?i)\b(?:api[-_]?key|access[-_]?token|client[-_]?secret|password)\b"
    r"[ \t]*[:=][ \t]*[\"']([^\"']+)[\"']"
)
BEARER_CREDENTIAL = re.compile(r"(?i)\bauthorization[ \t]*:[ \t]*bearer[ \t]+([^\s\"']+)")

SENSITIVE_CONTENT_PATTERNS: tuple[tuple[str, re.Pattern[bytes]], ...] = (
    (
        "macos_user_absolute_path",
        re.compile(rb"/Users/[A-Za-z0-9._-]+(?:/[^\x00\r\n\t \"'<>]*)?"),
    ),
    (
        "linux_user_absolute_path",
        re.compile(rb"/home/[A-Za-z0-9._-]+(?:/[^\x00\r\n\t \"'<>]*)?"),
    ),
    (
        "windows_user_absolute_path",
        re.compile(
            rb"[A-Za-z]:[\\/]+Users[\\/]+[A-Za-z0-9._-]+"
            rb"(?:[\\/]+[^\x00\r\n\t \"'<>]*)?",
            re.IGNORECASE,
        ),
    ),
    (
        "gpu_uuid",
        re.compile(
            rb"(?:GPU-|MIG-GPU-)[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
            rb"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
        ),
    ),
)

INFRASTRUCTURE_ASSIGNMENT = re.compile(
    r"^[ \t]*(?:export[ \t]+)?(?:RUNPOD_POD_ID|RUNPOD_MACHINE_ID|GPU_UUIDS?)"
    r"[ \t]*=[ \t]*([^\r\n]*?)[ \t]*$",
    re.MULTILINE,
)

SENSITIVE_INFRASTRUCTURE_KEYS = frozenset(
    {
        "device_uuid",
        "gpu_id",
        "gpu_ids",
        "gpu_uuid",
        "gpu_uuids",
        "machine_gpu_identity",
        "machine_id",
        "pod_id",
        "runpod_machine_id",
        "runpod_pod_id",
    }
)

PRIVATE_RESPONSE_BODY_KEYS = frozenset(
    {
        "api_response",
        "choices",
        "classifier_response",
        "final_response",
        "http_response_body",
        "judge_response",
        "provider_headers",
        "provider_response",
        "provider_response_body",
        "provider_response_id",
        "raw",
        "raw_output",
        "raw_outputs",
        "raw_response",
        "raw_response_body",
        "response",
        "response_body",
        "response_content",
        "response_text",
        "trajectory_response",
    }
)

PLACEHOLDER_VALUES = frozenset(
    {
        "",
        "''",
        '""',
        "changeme",
        "do-not-persist",
        "dummy",
        "example",
        "none",
        "null",
        "redacted",
        "test",
    }
)


class ReleaseAuditError(RuntimeError):
    """The proposed public release violates a fail-closed boundary."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalized_relative_path(value: str | Path) -> str:
    rendered = str(value).replace(os.sep, "/")
    path = PurePosixPath(rendered)
    if path.is_absolute() or not path.parts or ".." in path.parts or "." in path.parts:
        raise ReleaseAuditError(f"candidate path is not a normalized relative path: {value!s}")
    return path.as_posix()


def discover_release_candidates(root: Path) -> tuple[str, ...]:
    """Return tracked plus nonignored untracked files from a real Git worktree."""

    completed = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "ls-files",
            "-z",
            "--cached",
            "--others",
            "--exclude-standard",
        ],
        check=False,
        capture_output=True,
    )
    if completed.returncode:
        message = completed.stderr.decode("utf-8", errors="replace").strip()
        raise ReleaseAuditError(f"cannot enumerate Git release candidates: {message}")
    candidates = tuple(
        sorted(
            {
                _normalized_relative_path(raw.decode("utf-8", errors="strict"))
                for raw in completed.stdout.split(b"\0")
                if raw
            }
        )
    )
    if not candidates:
        raise ReleaseAuditError("Git returned no release candidates")
    return candidates


def _path_is_allowlisted(relative: str) -> bool:
    if relative in ROOT_ALLOWLIST:
        return True
    if relative in PUBLIC_MANIFEST_ALLOWLIST:
        return True
    if relative.startswith("data/manifests/"):
        return False
    suffix = PurePosixPath(relative).suffix.lower()
    return any(
        relative.startswith(prefix) and suffix in suffixes for prefix, suffixes in ALLOWED_TREES
    )


def _looks_like_placeholder(value: str) -> bool:
    rendered = value.strip().strip("\"'").strip()
    normalized = rendered.lower()
    return (
        normalized in PLACEHOLDER_VALUES
        or normalized.startswith("${")
        or normalized.startswith("$env:")
        or (normalized.startswith("<") and normalized.endswith(">"))
        or normalized.startswith("your_")
        or normalized.startswith("your-")
        or re.fullmatch(
            r"\$?[A-Z][A-Z0-9_]*(?:KEY|TOKEN|SECRET|PASSWORD)",
            rendered,
        )
        is not None
    )


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueKeySafeLoader,
    node: yaml.nodes.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as exc:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable mapping key",
                key_node.start_mark,
            ) from exc
        if duplicate:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeySafeLoader.add_constructor(
    BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _structured_payloads(relative: str, text: str) -> tuple[Any, ...]:
    """Parse release metadata without allowing duplicate-key scan bypasses."""

    suffix = PurePosixPath(relative).suffix.lower()
    try:
        if suffix == ".json":
            return (json.loads(text, object_pairs_hook=_strict_json_object),)
        if suffix == ".jsonl":
            return tuple(
                json.loads(line, object_pairs_hook=_strict_json_object)
                for line in text.splitlines()
                if line.strip()
            )
        if suffix in {".yaml", ".yml"}:
            return tuple(yaml.load_all(text, Loader=_UniqueKeySafeLoader))
    except (ConstructorError, json.JSONDecodeError, ValueError, yaml.YAMLError) as exc:
        raise ReleaseAuditError(
            f"structured release candidate is invalid or has duplicate keys: {relative}"
        ) from exc
    return ()


def _normalized_field_name(value: object) -> str:
    rendered = str(value)
    rendered = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", rendered)
    return re.sub(r"[^a-z0-9]+", "_", rendered.lower()).strip("_")


def _credential_field_name(field: str) -> bool:
    if field in {
        "access_token",
        "api_key",
        "auth_token",
        "authorization",
        "bearer_token",
        "client_secret",
        "credential",
        "credentials",
        "password",
        "passwd",
        "private_key",
        "secret",
    }:
        return True
    return field.endswith(
        (
            "_access_token",
            "_api_key",
            "_auth_token",
            "_client_secret",
            "_credential",
            "_password",
            "_private_key",
            "_secret",
        )
    )


def _structured_content_findings(payload: Any) -> list[str]:
    findings: list[str] = []

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for raw_key, child in value.items():
                field = _normalized_field_name(raw_key)
                if field in SENSITIVE_INFRASTRUCTURE_KEYS:
                    findings.append("sensitive_infrastructure_identifier")
                if field in PRIVATE_RESPONSE_BODY_KEYS:
                    findings.append("private_provider_response_body")
                if _credential_field_name(field):
                    if not isinstance(child, str) or not _looks_like_placeholder(child):
                        findings.append("populated_structured_credential")
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
        elif isinstance(value, str):
            encoded = value.encode("utf-8")
            for name, pattern in SENSITIVE_CONTENT_PATTERNS:
                if pattern.search(encoded):
                    findings.append(name)
            if _secret_findings(encoded, text=value):
                findings.append("embedded_secret_or_credential")
            for match in INFRASTRUCTURE_ASSIGNMENT.finditer(value):
                if not _looks_like_placeholder(match.group(1)):
                    findings.append("sensitive_infrastructure_assignment")
                    break

    visit(payload)
    return findings


def _sensitive_content_findings(
    data: bytes,
    *,
    text: str | None,
    relative: str,
) -> list[str]:
    findings = [name for name, pattern in SENSITIVE_CONTENT_PATTERNS if pattern.search(data)]
    if text is None:
        return findings
    for match in INFRASTRUCTURE_ASSIGNMENT.finditer(text):
        if not _looks_like_placeholder(match.group(1)):
            findings.append("sensitive_infrastructure_assignment")
            break
    for payload in _structured_payloads(relative, text):
        findings.extend(_structured_content_findings(payload))
    return findings


def _secret_findings(data: bytes, *, text: str | None) -> list[str]:
    findings = [name for name, pattern in TOKEN_PATTERNS if pattern.search(data)]
    if text is None:
        return findings
    for pattern_name, pattern in (
        ("populated_secret_environment_assignment", ENV_ASSIGNMENT),
        ("populated_credential_assignment", GENERIC_ASSIGNMENT),
        ("bearer_credential", BEARER_CREDENTIAL),
    ):
        for match in pattern.finditer(text):
            if not _looks_like_placeholder(match.group(1)):
                findings.append(pattern_name)
                break
    return findings


def audit_release(
    root: Path,
    candidates: Iterable[str | Path],
    *,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
    required_paths: frozenset[str] = REQUIRED_RELEASE_PATHS,
) -> dict[str, object]:
    """Audit an explicit candidate set and return a deterministic manifest."""

    root = root.resolve()
    if max_file_bytes <= 0:
        raise ValueError("max_file_bytes must be positive")
    normalized = tuple(sorted({_normalized_relative_path(path) for path in candidates}))
    if not normalized:
        raise ReleaseAuditError("release candidate set is empty")
    missing_required = sorted(required_paths.difference(normalized))
    if missing_required:
        raise ReleaseAuditError(f"required release files are missing: {missing_required}")
    casefolded: dict[str, str] = {}
    rows: list[dict[str, object]] = []
    total_bytes = 0
    for relative in normalized:
        folded = relative.casefold()
        previous = casefolded.get(folded)
        if previous is not None and previous != relative:
            raise ReleaseAuditError(
                f"case-colliding candidate paths are not portable: {previous!r}, {relative!r}"
            )
        casefolded[folded] = relative
        if relative.startswith(FORBIDDEN_PREFIXES):
            raise ReleaseAuditError(f"forbidden raw/cache/upstream path in release: {relative}")
        if PurePosixPath(relative).name in {".env", "credentials", "credentials.json"}:
            raise ReleaseAuditError(f"credential file is forbidden from release: {relative}")
        suffix = PurePosixPath(relative).suffix.lower()
        if suffix in FORBIDDEN_SUFFIXES:
            raise ReleaseAuditError(f"model/archive/key artifact is forbidden: {relative}")
        if not _path_is_allowlisted(relative):
            raise ReleaseAuditError(
                f"path is outside the explicit public-release allowlist: {relative}"
            )

        path = root / relative
        if path.is_symlink():
            raise ReleaseAuditError(f"symlinks are forbidden from release: {relative}")
        if not path.is_file():
            raise ReleaseAuditError(f"candidate is missing or not a regular file: {relative}")
        size = path.stat().st_size
        if size > max_file_bytes:
            raise ReleaseAuditError(
                f"candidate exceeds {max_file_bytes} bytes and needs explicit review: {relative}"
            )
        data = path.read_bytes()
        text: str | None = None
        if suffix in TEXT_SUFFIXES or relative in ROOT_ALLOWLIST:
            if b"\0" in data:
                raise ReleaseAuditError(f"text candidate contains NUL bytes: {relative}")
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ReleaseAuditError(f"text candidate is not valid UTF-8: {relative}") from exc
        secret_findings = _secret_findings(data, text=text)
        if secret_findings:
            raise ReleaseAuditError(
                f"possible secret in {relative}: {sorted(set(secret_findings))}"
            )
        sensitive_findings = _sensitive_content_findings(
            data,
            text=text,
            relative=relative,
        )
        if sensitive_findings:
            raise ReleaseAuditError(
                f"sensitive public-release content in {relative}: {sorted(set(sensitive_findings))}"
            )
        total_bytes += size
        rows.append({"path": relative, "size_bytes": size, "sha256": _sha256(path)})

    manifest: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "candidate_count": len(rows),
        "total_bytes": total_bytes,
        "max_file_bytes": max_file_bytes,
        "files": rows,
    }
    canonical = json.dumps(
        manifest,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    manifest["manifest_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--max-file-bytes", type=int, default=DEFAULT_MAX_FILE_BYTES)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        candidates = discover_release_candidates(args.root.resolve())
        result = audit_release(
            args.root,
            candidates,
            max_file_bytes=args.max_file_bytes,
        )
    except (OSError, UnicodeError, ValueError, ReleaseAuditError) as exc:
        print(f"release audit failed: {exc}", file=sys.stderr)
        return 2
    serialized = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
