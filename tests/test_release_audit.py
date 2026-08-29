from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "release_audit.py"
SPEC = importlib.util.spec_from_file_location("release_audit", SCRIPT)
assert SPEC and SPEC.loader
release_audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release_audit)


def _safe_release_tree(root: Path) -> list[str]:
    payloads = {
        ".gitignore": ".env\ndata/raw/\ndata/interim/\ndata/upstream/\n",
        "LICENSE": "MIT test fixture\n",
        "README.md": "# Reproducible test fixture\n",
        "THIRD_PARTY_NOTICES.md": "No third-party payloads are redistributed.\n",
        "config/preregistration.yaml": "frozen_before_primary_results: true\n",
        "src/model_forensics/example.py": "VALUE = 1\n",
        "tests/test_example.py": "def test_example():\n    assert True\n",
        ".env.example": "HF_TOKEN=\nOPENROUTER_API_KEY=\nRUNPOD_API_KEY=\n",
    }
    for relative, content in payloads.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return sorted(payloads)


def test_safe_explicit_release_returns_content_addressed_manifest(tmp_path: Path) -> None:
    candidates = _safe_release_tree(tmp_path)

    result = release_audit.audit_release(tmp_path, candidates)

    assert result["status"] == "passed"
    assert result["candidate_count"] == len(candidates)
    assert len(result["manifest_sha256"]) == 64
    assert {row["path"] for row in result["files"]} == set(candidates)


def test_secret_token_and_populated_environment_assignment_fail_closed(tmp_path: Path) -> None:
    candidates = _safe_release_tree(tmp_path)
    source = tmp_path / "src/model_forensics/example.py"
    # Materialize a token without placing a scanner-shaped credential in this test source.
    source.write_text("LEAK = " + "sk-or-v1-" + "A" * 32 + "\n", encoding="utf-8")
    with pytest.raises(release_audit.ReleaseAuditError, match="possible secret"):
        release_audit.audit_release(tmp_path, candidates)

    source.write_text("VALUE = 1\n", encoding="utf-8")
    env_file = tmp_path / ".env.example"
    env_file.write_text("HF_TOKEN=" + "hf_" + "B" * 32 + "\n", encoding="utf-8")
    with pytest.raises(release_audit.ReleaseAuditError, match="possible secret"):
        release_audit.audit_release(tmp_path, candidates)

    env_file.write_text("RUNPOD_API_KEY=plain-but-still-secret\n", encoding="utf-8")
    with pytest.raises(release_audit.ReleaseAuditError, match="possible secret"):
        release_audit.audit_release(tmp_path, candidates)


@pytest.mark.parametrize(
    "relative",
    [
        "data/raw/rollouts.jsonl",
        "data/interim/resampling.jsonl",
        "data/upstream/value-leakage/source.py",
        "reports/staging/draft.md",
    ],
)
def test_raw_staging_and_upstream_paths_are_forbidden(tmp_path: Path, relative: str) -> None:
    candidates = _safe_release_tree(tmp_path)
    path = tmp_path / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("payload\n", encoding="utf-8")

    with pytest.raises(release_audit.ReleaseAuditError, match="forbidden"):
        release_audit.audit_release(tmp_path, [*candidates, relative])


def test_symlink_and_unknown_tree_require_explicit_release_review(tmp_path: Path) -> None:
    candidates = _safe_release_tree(tmp_path)
    link = tmp_path / "docs/link.md"
    link.parent.mkdir(parents=True)
    link.symlink_to(tmp_path / "README.md")
    with pytest.raises(release_audit.ReleaseAuditError, match="symlinks"):
        release_audit.audit_release(tmp_path, [*candidates, "docs/link.md"])

    unknown = tmp_path / "notebooks/analysis.ipynb"
    unknown.parent.mkdir()
    unknown.write_text("{}\n", encoding="utf-8")
    with pytest.raises(release_audit.ReleaseAuditError, match="outside"):
        release_audit.audit_release(tmp_path, [*candidates, "notebooks/analysis.ipynb"])


@pytest.mark.parametrize(
    "relative",
    [
        "data/manifests/unreviewed.json",
        "data/manifests/gpu_environment.json",
        "data/manifests/gpu_preflight.json",
        "data/manifests/paid_run_approval.json",
        "data/manifests/runpod_watchdog.json",
    ],
)
def test_manifest_tree_is_exactly_curated_not_suffix_allowlisted(
    tmp_path: Path,
    relative: str,
) -> None:
    candidates = _safe_release_tree(tmp_path)
    path = tmp_path / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"schema_version":1}\n', encoding="utf-8")

    with pytest.raises(release_audit.ReleaseAuditError, match="explicit public-release allowlist"):
        release_audit.audit_release(tmp_path, [*candidates, relative])


@pytest.mark.parametrize(
    "private_path",
    [
        "/" + "Users" + "/alice/private/output.json",
        "/" + "home" + "/runner/private/output.json",
        "C:" + "\\" + "Users" + "\\alice\\private\\output.json",
    ],
)
def test_local_absolute_user_paths_fail_closed(tmp_path: Path, private_path: str) -> None:
    candidates = _safe_release_tree(tmp_path)
    relative = "docs/leak.md"
    path = tmp_path / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"artifact: {private_path}\n", encoding="utf-8")

    with pytest.raises(release_audit.ReleaseAuditError, match="user_absolute_path"):
        release_audit.audit_release(tmp_path, [*candidates, relative])


def test_escaped_structured_local_path_cannot_bypass_scan(tmp_path: Path) -> None:
    candidates = _safe_release_tree(tmp_path)
    relative = "data/manifests/sampling_manifest.json"
    path = tmp_path / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    escaped_path = "\\u002fUsers\\u002falice\\u002fprivate.json"
    path.write_text('{"artifact":"' + escaped_path + '"}\n', encoding="utf-8")

    with pytest.raises(release_audit.ReleaseAuditError, match="macos_user_absolute_path"):
        release_audit.audit_release(tmp_path, [*candidates, relative])


def test_runpod_pod_and_gpu_identifiers_fail_closed(tmp_path: Path) -> None:
    candidates = _safe_release_tree(tmp_path)
    manifest_relative = "data/manifests/sampling_manifest.json"
    manifest = tmp_path / manifest_relative
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps({"schema_version": 1, "pod_id": "p0d9t4q2z8r6w3x1"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(
        release_audit.ReleaseAuditError,
        match="sensitive_infrastructure_identifier",
    ):
        release_audit.audit_release(tmp_path, [*candidates, manifest_relative])

    docs_relative = "docs/gpu-leak.md"
    docs = tmp_path / docs_relative
    docs.parent.mkdir(parents=True, exist_ok=True)
    gpu_identifier = "GPU-" + "12345678-1234-1234-1234-123456789abc"
    docs.write_text(f"device: {gpu_identifier}\n", encoding="utf-8")
    with pytest.raises(release_audit.ReleaseAuditError, match="gpu_uuid"):
        release_audit.audit_release(tmp_path, [*candidates, docs_relative])

    docs.write_text(
        "RUNPOD_" + "POD_ID=" + "p0d9t4q2z8r6w3x1\n",
        encoding="utf-8",
    )
    with pytest.raises(
        release_audit.ReleaseAuditError,
        match="sensitive_infrastructure_assignment",
    ):
        release_audit.audit_release(tmp_path, [*candidates, docs_relative])


@pytest.mark.parametrize(
    "private_field",
    ["raw_response", "response_body", "provider_response", "final_response", "choices"],
)
def test_private_provider_response_bodies_fail_closed(
    tmp_path: Path,
    private_field: str,
) -> None:
    candidates = _safe_release_tree(tmp_path)
    relative = "data/manifests/sampling_manifest.json"
    path = tmp_path / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"schema_version": 1, private_field: {"content": "private judgment"}}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(release_audit.ReleaseAuditError, match="private_provider_response_body"):
        release_audit.audit_release(tmp_path, [*candidates, relative])


def test_structured_credentials_are_rejected_but_env_names_and_hashes_are_safe(
    tmp_path: Path,
) -> None:
    candidates = _safe_release_tree(tmp_path)
    relative = "config/private.yaml"
    path = tmp_path / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("openrouter_api_key: live-credential-value\n", encoding="utf-8")
    with pytest.raises(release_audit.ReleaseAuditError, match="populated_structured_credential"):
        release_audit.audit_release(tmp_path, [*candidates, relative])

    path.write_text(
        "openrouter_api_key: OPENROUTER_API_KEY\n"
        "provider_response_id_hash: sha256:" + "a" * 64 + "\nraw_path: data/raw/private.jsonl\n",
        encoding="utf-8",
    )
    result = release_audit.audit_release(tmp_path, [*candidates, relative])
    assert result["status"] == "passed"


def test_secret_assignment_embedded_in_structured_text_fails_closed(tmp_path: Path) -> None:
    candidates = _safe_release_tree(tmp_path)
    relative = "data/manifests/sampling_manifest.json"
    path = tmp_path / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    embedded = "OPENROUTER_" + "API_KEY=live-credential-value"
    path.write_text(json.dumps({"log": embedded}) + "\n", encoding="utf-8")

    with pytest.raises(release_audit.ReleaseAuditError, match="embedded_secret_or_credential"):
        release_audit.audit_release(tmp_path, [*candidates, relative])


def test_duplicate_structured_keys_cannot_bypass_content_scan(tmp_path: Path) -> None:
    candidates = _safe_release_tree(tmp_path)
    relative = "data/manifests/sampling_manifest.json"
    path = tmp_path / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"schema_version":1,"schema_version":1}\n', encoding="utf-8")

    with pytest.raises(release_audit.ReleaseAuditError, match="duplicate keys"):
        release_audit.audit_release(tmp_path, [*candidates, relative])


def test_git_discovery_includes_nonignored_untracked_and_excludes_ignored(tmp_path: Path) -> None:
    candidates = _safe_release_tree(tmp_path)
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    ignored = tmp_path / "data/raw/private.jsonl"
    ignored.parent.mkdir(parents=True)
    ignored.write_text("private\n", encoding="utf-8")

    discovered = release_audit.discover_release_candidates(tmp_path)

    assert set(candidates).issubset(discovered)
    assert "data/raw/private.jsonl" not in discovered


def test_current_git_release_candidates_pass() -> None:
    root = SCRIPT.parents[1]
    candidates = release_audit.discover_release_candidates(root)

    result = release_audit.audit_release(root, candidates)

    assert result["status"] == "passed"
