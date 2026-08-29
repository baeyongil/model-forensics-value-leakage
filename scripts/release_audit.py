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
import stat
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path, PurePosixPath
from typing import Any, NamedTuple

import yaml
from yaml.constructor import ConstructorError
from yaml.resolver import BaseResolver

from model_forensics.io import stable_hash
from model_forensics.public_results import (
    BEHAVIOR_FIELDS,
    EFFECT_FIELDS,
    FIGURE_PATHS,
    LENS_FIELDS,
    PublicResultsError,
    render_release_figures,
    sanitize_behavior_rows,
    sanitize_effect_rows,
    validate_lens_aggregate,
    validate_released_evidence,
)

SCHEMA_VERSION = 2
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
)

PRIMARY_FIGURE_ALLOWLIST = frozenset(
    f"reports/figures/{filename}" for filename in FIGURE_PATHS.values()
)
SMOKE_FIGURE_SHA256 = {
    "reports/figures/smoke/first_vs_final_bias.png": (
        "4d8cb56b71f70161a39ab92e9d587e839775b5ba3d658f194a7035adb43f5601"
    ),
    "reports/figures/smoke/lens_layer_position_heatmap.png": (
        "60f4d2f1f6edae4e3266fc59ca06e65671f5c4f7630abc700c5fc90dfa39b99d"
    ),
    "reports/figures/smoke/sentence_causal_effect_forest.png": (
        "e55098b796da1d80176183ed3d62042a512138c5922e590f7d0e3f7cd3bdf12c"
    ),
}
PUBLIC_FIGURE_ALLOWLIST = PRIMARY_FIGURE_ALLOWLIST.union(SMOKE_FIGURE_SHA256)

PUBLIC_RESULTS_ALLOWLIST = frozenset(
    {
        "reports/results/README.md",
        "reports/results/released_evidence.json",
        "reports/results/results_manifest.json",
        "reports/results/tables/behavior_stage_summary.jsonl",
        "reports/results/tables/sentence_effects.jsonl",
        "reports/results/tables/lens_direction_heatmap.jsonl",
    }
)
PUBLIC_RESULTS_BUNDLE_PATHS = PUBLIC_RESULTS_ALLOWLIST.difference(
    {"reports/results/README.md"}
)

# Every public manifest is reviewed by exact path.  In particular, this does
# not include RunPod watchdog/preflight/environment state, paid-response
# checkpoints, or any raw/interim artifact.  Adding a newly generated manifest
# to the public repository therefore requires an explicit code review here;
# merely giving it a JSON/YAML suffix is never sufficient.
PUBLIC_MANIFEST_ALLOWLIST = frozenset(
    {
        # Only already-reviewed static ledgers/probe metadata and deterministic
        # synthetic smoke artifacts are public. Every primary runtime manifest
        # remains ignored/private; public evidence is aggregate-only.
        "data/manifests/cost_ledger.yaml",
        "data/manifests/lens_probe_token_verification.json",
        "data/manifests/smoke/adjudication_manifest.jsonl",
        "data/manifests/smoke/anchor_manifest.json",
        "data/manifests/smoke/sampling_manifest.json",
        "data/manifests/smoke/smoke_completion.json",
        "data/manifests/time_ledger.yaml",
        "data/manifests/upstream_reference.json",
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

# Match an SSH invocation anywhere in prose, including Markdown bullets and
# inline-code spans.  Anchoring this at the start of a line leaves a trivial
# release-scanner bypass such as ``- `ssh root@literal-host -p 1234` ``.
SSH_COMMAND_LINE = re.compile(
    r"(?im)(?<![A-Za-z0-9_-])ssh[ \t]+"
    r"(?=[^\r\n]*(?:[A-Za-z0-9._-]+@|(?:[0-9]{1,3}\.){3}[0-9]{1,3}|\[[0-9a-f:]+\]))"
    r"[^\r\n]*"
)
RAW_POD_LABEL = re.compile(
    r"(?i)\b(?:runpod[ _-]+)?pod[ _-]?id\b[ \t]*(?:=|:)[ \t]*"
    r"([A-Za-z0-9][A-Za-z0-9_-]{5,127})"
)
RAW_POD_API_PATH = re.compile(
    r"(?i)\b(?:rest\.)?runpod\.(?:io|net)/(?:v[12]/)?pods/"
    r"([A-Za-z0-9][A-Za-z0-9_-]{5,127})"
)
# Also reject an opaque identifier introduced directly as a Pod value in prose
# (for example ``Pod `p0d9t4q2z8r6w3x1` ``), without treating ordinary phrases
# such as "Pod lifecycle" as identifiers.  Real opaque IDs must contain a
# digit; placeholders are handled separately by ``_looks_like_placeholder``.
RAW_POD_PROSE = re.compile(
    r"(?i)\b(?:runpod[ _-]+)?pod\b[ \t]+"
    r"(?:id(?:entifier)?[ \t]*(?:=|:)?[ \t]*)?"
    r"(?:is[ \t]+)?[`\"']?"
    r"((?=[A-Za-z0-9_-]{6,128}(?:[`\"']|\b))"
    r"(?=[A-Za-z0-9_-]*[0-9])"
    r"[A-Za-z0-9][A-Za-z0-9_-]{5,127})"
    r"[`\"']?"
)
RAW_SESSION_LABEL = re.compile(
    r"(?i)\b(?:gpu[ _-]budget[ _-])?session[ _-]?id\b"
    r"[ \t]*(?:=|:)[ \t]*([A-Za-z0-9][A-Za-z0-9_-]{7,255})"
)
# The documented production nonce is generated by ``openssl rand -hex 32``.
# Detect that concrete 64-hex value when it is introduced as a session value
# anywhere in prose (including Markdown bullets and inline-code spans), while
# leaving ordinary descriptions and explicit placeholders alone.
RAW_SESSION_PROSE = re.compile(
    r"(?i)\b(?:gpu[ _-]+budget[ _-]+)?(?:gpu[ _-]+)?session"
    r"(?:[ _-]+(?:id(?:entifier)?|nonce))?\b[ \t]*"
    r"(?:(?:is|was)[ \t]+|(?:=|:)[ \t]*)?"
    r"[`\"']?([0-9a-f]{64})[`\"']?"
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
        "phase-session-opaque",
        "session-placeholder",
        "test",
    }
)


class ReleaseAuditError(RuntimeError):
    """The proposed public release violates a fail-closed boundary."""


class GitIndexEntry(NamedTuple):
    """One stage-zero Git index blob, independent of working-tree bytes."""

    mode: str
    object_id: str
    data: bytes


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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


def discover_git_index_entries(root: Path) -> dict[str, GitIndexEntry]:
    """Read stage-zero tracked blobs without trusting their working-tree copies."""

    completed = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-s", "-z"],
        check=False,
        capture_output=True,
    )
    if completed.returncode:
        message = completed.stderr.decode("utf-8", errors="replace").strip()
        raise ReleaseAuditError(f"cannot enumerate Git index entries: {message}")
    parsed: dict[str, tuple[str, str]] = {}
    for raw in completed.stdout.split(b"\0"):
        if not raw:
            continue
        try:
            header, raw_path = raw.split(b"\t", 1)
            mode, object_id, stage = header.decode("ascii").split(" ")
            relative = _normalized_relative_path(raw_path.decode("utf-8", errors="strict"))
        except (UnicodeError, ValueError) as exc:
            raise ReleaseAuditError("Git returned a malformed index entry") from exc
        if stage != "0":
            raise ReleaseAuditError(f"unmerged Git index entry is forbidden: {relative}")
        if relative in parsed:
            raise ReleaseAuditError(f"duplicate Git index entry is forbidden: {relative}")
        parsed[relative] = (mode, object_id)

    object_cache: dict[str, bytes] = {}
    entries: dict[str, GitIndexEntry] = {}
    for relative, (mode, object_id) in parsed.items():
        data = object_cache.get(object_id)
        if data is None:
            blob = subprocess.run(
                ["git", "-C", str(root), "cat-file", "blob", object_id],
                check=False,
                capture_output=True,
            )
            if blob.returncode:
                raise ReleaseAuditError(f"cannot read Git index blob for {relative}")
            data = blob.stdout
            object_cache[object_id] = data
        entries[relative] = GitIndexEntry(mode=mode, object_id=object_id, data=data)
    return entries


def _path_is_allowlisted(relative: str) -> bool:
    if relative in ROOT_ALLOWLIST:
        return True
    if relative in PUBLIC_MANIFEST_ALLOWLIST:
        return True
    if relative in PUBLIC_RESULTS_ALLOWLIST:
        return True
    if relative in PUBLIC_FIGURE_ALLOWLIST:
        return True
    if relative.startswith("reports/results/"):
        return False
    if relative.startswith("data/manifests/"):
        return False
    suffix = PurePosixPath(relative).suffix.lower()
    return any(
        relative.startswith(prefix) and suffix in suffixes for prefix, suffixes in ALLOWED_TREES
    )


def _secure_worktree_path(root: Path, relative: str) -> Path:
    """Reject every symlink component and every resolved repository escape."""

    path = root
    for component in PurePosixPath(relative).parts:
        path = path / component
        if path.is_symlink():
            raise ReleaseAuditError(f"symlinks are forbidden from release: {relative}")
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(root)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise ReleaseAuditError(
            f"candidate is missing or escapes the release root: {relative}"
        ) from exc
    details = path.lstat()
    if not stat.S_ISREG(details.st_mode):
        raise ReleaseAuditError(f"candidate is missing or not a regular file: {relative}")
    return resolved


def _looks_like_placeholder(value: str) -> bool:
    rendered = value.strip().strip("\"'").strip()
    normalized = rendered.lower()
    return (
        normalized in PLACEHOLDER_VALUES
        or normalized.startswith("${")
        or normalized.startswith("$env:")
        or (normalized.startswith("<") and normalized.endswith(">"))
        or (normalized.startswith("{") and normalized.endswith("}"))
        or normalized.startswith("your_")
        or normalized.startswith("your-")
        or re.fullmatch(
            r"\$?[A-Z][A-Z0-9_]*(?:KEY|TOKEN|SECRET|PASSWORD)",
            rendered,
        )
        is not None
    )


def _infrastructure_prose_findings(text: str) -> list[str]:
    """Detect literal connection/session material while permitting placeholders."""

    findings: list[str] = []
    for match in SSH_COMMAND_LINE.finditer(text):
        command = re.sub(r"^[ \t]*\$[ \t]+", "", match.group(0))
        placeholder = re.search(
            r"(?:\$\{?[A-Z][A-Z0-9_]*\}?|<[^>]+>|\{[^}]+\}|\[[A-Z_-]+\])",
            command,
            re.IGNORECASE,
        )
        if placeholder is None:
            findings.append("literal_ssh_command_or_endpoint")
            break
    for name, pattern in (
        ("raw_runpod_pod_identifier", RAW_POD_LABEL),
        ("raw_runpod_pod_identifier", RAW_POD_API_PATH),
        ("raw_runpod_pod_identifier", RAW_POD_PROSE),
        ("raw_gpu_session_identifier", RAW_SESSION_LABEL),
        ("raw_gpu_session_identifier", RAW_SESSION_PROSE),
    ):
        for match in pattern.finditer(text):
            if not _looks_like_placeholder(match.group(1)):
                findings.append(name)
                break
    return findings


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
            findings.extend(_infrastructure_prose_findings(value))

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
    if PurePosixPath(relative).suffix.lower() in {".md", ".txt"}:
        findings.extend(_infrastructure_prose_findings(text))
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


def _exact_public_fields(
    row: Mapping[str, Any], fields: Iterable[str], *, label: str
) -> None:
    expected = set(fields)
    if set(row) != expected:
        missing = sorted(expected.difference(row))
        extra = sorted(set(row).difference(expected))
        raise ReleaseAuditError(
            f"public result {label} fields changed; missing={missing}, extra={extra}"
        )


def _decoded_public_text(read_bytes: Callable[[str], bytes], relative: str) -> str:
    try:
        return read_bytes(relative).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ReleaseAuditError(f"public result artifact is not UTF-8: {relative}") from exc


def _public_json_object(
    read_bytes: Callable[[str], bytes], relative: str
) -> dict[str, Any]:
    payloads = _structured_payloads(
        relative,
        _decoded_public_text(read_bytes, relative),
    )
    if len(payloads) != 1 or not isinstance(payloads[0], dict):
        raise ReleaseAuditError(f"public result artifact is not one JSON object: {relative}")
    return payloads[0]


def _public_jsonl_rows(
    read_bytes: Callable[[str], bytes], relative: str
) -> list[dict[str, Any]]:
    payloads = _structured_payloads(
        relative,
        _decoded_public_text(read_bytes, relative),
    )
    if not all(isinstance(payload, dict) for payload in payloads):
        raise ReleaseAuditError(f"public result table has a non-object row: {relative}")
    return list(payloads)


def _audit_public_results_bundle(
    candidates: Iterable[str],
    *,
    read_bytes: Callable[[str], bytes],
    source_label: str,
) -> None:
    """Authenticate the exact aggregate-only public bundle when it is present."""

    candidate_set = set(candidates)
    present = PUBLIC_RESULTS_BUNDLE_PATHS.intersection(candidate_set)
    if not present:
        return
    missing = sorted(PUBLIC_RESULTS_BUNDLE_PATHS.difference(present))
    if missing:
        raise ReleaseAuditError(f"public result bundle is incomplete: {missing}")

    evidence_relative = "reports/results/released_evidence.json"
    manifest_relative = "reports/results/results_manifest.json"
    evidence_payload = _public_json_object(read_bytes, evidence_relative)
    try:
        evidence = validate_released_evidence(evidence_payload)
    except PublicResultsError as exc:
        raise ReleaseAuditError(f"public result evidence schema failed: {exc}") from exc

    table_specs = {
        "behavior_stage_summary": (
            "reports/results/tables/behavior_stage_summary.jsonl",
            BEHAVIOR_FIELDS,
            sanitize_behavior_rows,
        ),
        "sentence_effects": (
            "reports/results/tables/sentence_effects.jsonl",
            EFFECT_FIELDS,
            sanitize_effect_rows,
        ),
        "lens_direction_heatmap": (
            "reports/results/tables/lens_direction_heatmap.jsonl",
            LENS_FIELDS,
            validate_lens_aggregate,
        ),
    }
    table_rows: dict[str, list[dict[str, Any]]] = {}
    for name, (relative, fields, validator) in table_specs.items():
        rows = _public_jsonl_rows(read_bytes, relative)
        for index, row in enumerate(rows, start=1):
            _exact_public_fields(row, fields, label=f"{name} row {index}")
        try:
            canonical_rows = validator(rows)
        except PublicResultsError as exc:
            raise ReleaseAuditError(f"public result {name} schema failed: {exc}") from exc
        if canonical_rows != rows or rows != evidence[name]:
            raise ReleaseAuditError(
                f"public result {name} table differs from released evidence"
            )
        table_rows[name] = rows

    manifest = _public_json_object(read_bytes, manifest_relative)
    _exact_public_fields(
        manifest,
        {
            "schema_version",
            "status",
            "evidence",
            "aggregate_tables",
            "figure_outputs",
            "figure_hash_policy",
            "record_hash",
        },
        label="manifest",
    )
    if manifest.get("schema_version") != 1 or manifest.get(
        "status"
    ) != "authenticated_aggregate_release":
        raise ReleaseAuditError("public result manifest identity changed")
    if manifest.get("figure_hash_policy") != "regenerated_not_cross_platform_bitwise_pinned":
        raise ReleaseAuditError("public result figure hash policy changed")
    if manifest.get("record_hash") != stable_hash(
        {key: value for key, value in manifest.items() if key != "record_hash"}
    ):
        raise ReleaseAuditError("public result manifest record hash mismatch")

    evidence_link = manifest.get("evidence")
    if not isinstance(evidence_link, dict):
        raise ReleaseAuditError("public result manifest evidence link is not an object")
    _exact_public_fields(
        evidence_link,
        {"path", "sha256", "record_hash"},
        label="manifest evidence link",
    )
    if (
        evidence_link.get("path") != "released_evidence.json"
        or evidence_link.get("sha256") != _sha256_bytes(read_bytes(evidence_relative))
        or evidence_link.get("record_hash") != evidence["record_hash"]
    ):
        raise ReleaseAuditError("public result evidence link is not content-authenticated")

    aggregates = manifest.get("aggregate_tables")
    if not isinstance(aggregates, dict) or set(aggregates) != set(table_specs):
        raise ReleaseAuditError("public result aggregate-table inventory changed")
    for name, (relative, fields, _validator) in table_specs.items():
        metadata = aggregates[name]
        if not isinstance(metadata, dict):
            raise ReleaseAuditError(f"public result {name} metadata is not an object")
        _exact_public_fields(
            metadata,
            {"path", "sha256", "row_count", "fields"},
            label=f"manifest {name} metadata",
        )
        expected_relative = relative.removeprefix("reports/results/")
        if (
            metadata.get("path") != expected_relative
            or metadata.get("sha256") != _sha256_bytes(read_bytes(relative))
            or metadata.get("row_count") != len(table_rows[name])
            or metadata.get("fields") != list(fields)
        ):
            raise ReleaseAuditError(f"public result {name} metadata/hash mismatch")

    figures = manifest.get("figure_outputs")
    expected_figures = {
        "first_vs_final_bias": "reports/figures/first_vs_final_bias.png",
        "sentence_causal_effect_forest": (
            "reports/figures/sentence_causal_effect_forest.png"
        ),
    }
    if evidence["lens_direction_heatmap"]:
        expected_figures["lens_layer_position_heatmap"] = (
            "reports/figures/lens_layer_position_heatmap.png"
        )
    if figures != expected_figures:
        raise ReleaseAuditError("public result figure-output inventory/path changed")

    expected_paths = set(expected_figures.values())
    present_primary_figures = PRIMARY_FIGURE_ALLOWLIST.intersection(candidate_set)
    missing_figures = sorted(expected_paths.difference(present_primary_figures))
    unexpected_figures = sorted(present_primary_figures.difference(expected_paths))
    if missing_figures or unexpected_figures:
        raise ReleaseAuditError(
            f"public result figure inventory differs in {source_label}: "
            f"missing={missing_figures}, unexpected={unexpected_figures}"
        )
    with tempfile.TemporaryDirectory(prefix="model-forensics-release-figures-") as temporary:
        regenerated = render_release_figures(
            project_root=Path(temporary),
            evidence=evidence,
            figure_dir=Path(temporary),
        )
        for name, relative in expected_figures.items():
            regenerated_path = regenerated[name]
            if read_bytes(relative) != regenerated_path.read_bytes():
                raise ReleaseAuditError(
                    f"public result figure differs from trusted regeneration in {source_label}: "
                    f"{relative}"
                )


def _audit_candidate_bytes(
    *,
    relative: str,
    data: bytes,
    source_label: str,
    max_file_bytes: int,
) -> None:
    if len(data) > max_file_bytes:
        raise ReleaseAuditError(
            f"candidate exceeds {max_file_bytes} bytes in {source_label} and needs explicit "
            f"review: {relative}"
        )
    suffix = PurePosixPath(relative).suffix.lower()
    text: str | None = None
    if suffix in TEXT_SUFFIXES or relative in ROOT_ALLOWLIST:
        if b"\0" in data:
            raise ReleaseAuditError(
                f"text candidate contains NUL bytes in {source_label}: {relative}"
            )
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ReleaseAuditError(
                f"text candidate is not valid UTF-8 in {source_label}: {relative}"
            ) from exc
    secret_findings = _secret_findings(data, text=text)
    if secret_findings:
        raise ReleaseAuditError(
            f"possible secret in {source_label} {relative}: {sorted(set(secret_findings))}"
        )
    sensitive_findings = _sensitive_content_findings(
        data,
        text=text,
        relative=relative,
    )
    if sensitive_findings:
        raise ReleaseAuditError(
            f"sensitive public-release content in {source_label} {relative}: "
            f"{sorted(set(sensitive_findings))}"
        )


def _audit_smoke_figure_set(
    candidates: Iterable[str],
    *,
    read_bytes: Callable[[str], bytes],
    source_label: str,
) -> None:
    candidate_set = set(candidates)
    present = set(SMOKE_FIGURE_SHA256).intersection(candidate_set)
    if present and present != set(SMOKE_FIGURE_SHA256):
        missing = sorted(set(SMOKE_FIGURE_SHA256).difference(present))
        raise ReleaseAuditError(
            f"synthetic smoke figure set is incomplete in {source_label}: {missing}"
        )
    for relative in present:
        if _sha256_bytes(read_bytes(relative)) != SMOKE_FIGURE_SHA256[relative]:
            raise ReleaseAuditError(
                f"synthetic smoke figure hash mismatch in {source_label}: {relative}"
            )


def audit_release(
    root: Path,
    candidates: Iterable[str | Path],
    *,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
    required_paths: frozenset[str] = REQUIRED_RELEASE_PATHS,
    git_index_entries: Mapping[str, GitIndexEntry] | None = None,
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
    index_total_bytes = 0
    worktree_bytes: dict[str, bytes] = {}
    index_entries = dict(git_index_entries or {})
    unknown_index_paths = sorted(set(index_entries).difference(normalized))
    if unknown_index_paths:
        raise ReleaseAuditError(
            f"Git index contains paths absent from the candidate set: {unknown_index_paths}"
        )
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

        path = _secure_worktree_path(root, relative)
        data = path.read_bytes()
        _audit_candidate_bytes(
            relative=relative,
            data=data,
            source_label="working tree",
            max_file_bytes=max_file_bytes,
        )
        worktree_bytes[relative] = data
        total_bytes += len(data)
        row: dict[str, object] = {
            "path": relative,
            "size_bytes": len(data),
            "sha256": _sha256_bytes(data),
        }
        index_entry = index_entries.get(relative)
        if index_entry is not None:
            if index_entry.mode not in {"100644", "100755"}:
                raise ReleaseAuditError(
                    f"Git index entry is not a regular file: {relative}"
                )
            _audit_candidate_bytes(
                relative=relative,
                data=index_entry.data,
                source_label="Git index",
                max_file_bytes=max_file_bytes,
            )
            index_total_bytes += len(index_entry.data)
            row.update(
                {
                    "git_index_mode": index_entry.mode,
                    "git_index_size_bytes": len(index_entry.data),
                    "git_index_sha256": _sha256_bytes(index_entry.data),
                    "git_index_matches_worktree": index_entry.data == data,
                }
            )
        rows.append(row)

    worktree_candidates = set(normalized)
    primary_figures = PRIMARY_FIGURE_ALLOWLIST.intersection(worktree_candidates)
    if primary_figures and not PUBLIC_RESULTS_BUNDLE_PATHS.intersection(worktree_candidates):
        raise ReleaseAuditError("primary figures require the authenticated aggregate result bundle")
    _audit_smoke_figure_set(
        worktree_candidates,
        read_bytes=worktree_bytes.__getitem__,
        source_label="working tree",
    )
    _audit_public_results_bundle(
        worktree_candidates,
        read_bytes=worktree_bytes.__getitem__,
        source_label="working tree",
    )

    if index_entries:
        index_candidates = set(index_entries)
        primary_index_figures = PRIMARY_FIGURE_ALLOWLIST.intersection(index_candidates)
        if primary_index_figures and not PUBLIC_RESULTS_BUNDLE_PATHS.intersection(index_candidates):
            raise ReleaseAuditError(
                "Git-index primary figures require the authenticated aggregate result bundle"
            )
        def index_reader(relative: str) -> bytes:
            return index_entries[relative].data
        _audit_smoke_figure_set(
            index_candidates,
            read_bytes=index_reader,
            source_label="Git index",
        )
        _audit_public_results_bundle(
            index_candidates,
            read_bytes=index_reader,
            source_label="Git index",
        )

    manifest: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "candidate_count": len(rows),
        "total_bytes": total_bytes,
        "git_index_candidate_count": len(index_entries),
        "git_index_total_bytes": index_total_bytes,
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
        index_entries = discover_git_index_entries(args.root.resolve())
        result = audit_release(
            args.root,
            candidates,
            max_file_bytes=args.max_file_bytes,
            git_index_entries=index_entries,
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
