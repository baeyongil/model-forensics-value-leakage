from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

from model_forensics.public_results import (
    build_released_evidence,
    render_release_figures,
    write_release_bundle,
)

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


def _write_public_result_fixture(root: Path) -> list[str]:
    evidence = build_released_evidence(
        profile="qwen35_122b_primary",
        analysis_hash="sha256:" + "a" * 64,
        source_analysis_summary_sha256="b" * 64,
        lens_evidence_status="unavailable_not_zero",
        behavior_rows=[
            {
                "task": "giraffe",
                "condition": "baseline",
                "stage": stage,
                "rate": 0.5,
                "ci_low": 0.3,
                "ci_high": 0.7,
                "n": 10,
                "n_total": 10,
                "n_missing": 0,
                "missing_rate": 0.0,
                "good_side_rate": None,
                "good_side_ci_low": None,
                "good_side_ci_high": None,
                "good_side_n": 0,
                "signed_log_ratio_mean": None,
                "signed_log_ratio_median": None,
                "signed_log_ratio_n": 0,
                "signed_log_definition": "direction * log(estimate / threshold)",
            }
            for stage in ("first", "final")
        ],
        effect_rows=[
            {
                "sentence_class": "accuracy_commitment",
                "direction": "pooled",
                "contrast_id": "accuracy_commitment:pooled",
                "estimand": (
                    "equal-base-trace-weighted P(good side | retain) "
                    "- P(good side | divergent resample)"
                ),
                "estimate": None,
                "ci_low": None,
                "ci_high": None,
                "conclusion": "inconclusive",
                "clusters": 0,
                "total_clusters": 8,
                "complete_case_clusters": 0,
                "p_value": None,
                "p_value_adjusted": None,
                "inference_tier": "confirmatory",
                "is_confirmatory": True,
                "analysis_population": "base_trace_complete_case",
                "worst_case_bound": -1.0,
                "best_case_bound": 1.0,
                "divergent_coverage_gate_passed": True,
            }
        ],
        lens_rows=[],
    )
    write_release_bundle(
        project_root=root,
        results_dir=root / "reports/results",
        figure_dir=Path("reports/figures"),
        evidence=evidence,
    )
    render_release_figures(
        project_root=root,
        evidence=evidence,
        figure_dir=root / "reports/figures",
    )
    return sorted(
        str(path.relative_to(root))
        for directory in (root / "reports/results", root / "reports/figures")
        for path in directory.rglob("*")
        if path.is_file() and "smoke" not in path.parts
    )


def test_safe_explicit_release_returns_content_addressed_manifest(tmp_path: Path) -> None:
    candidates = _safe_release_tree(tmp_path)

    result = release_audit.audit_release(tmp_path, candidates)

    assert result["status"] == "passed"
    assert result["candidate_count"] == len(candidates)
    assert len(result["manifest_sha256"]) == 64
    assert {row["path"] for row in result["files"]} == set(candidates)


def test_public_result_bundle_has_exact_fields_and_machine_checked_hashes(
    tmp_path: Path,
) -> None:
    candidates = _safe_release_tree(tmp_path)
    result_paths = _write_public_result_fixture(tmp_path)

    result = release_audit.audit_release(tmp_path, [*candidates, *result_paths])

    assert result["status"] == "passed"
    assert "reports/results/results_manifest.json" in {
        row["path"] for row in result["files"]
    }

    table = tmp_path / "reports/results/tables/behavior_stage_summary.jsonl"
    rows = [json.loads(line) for line in table.read_text(encoding="utf-8").splitlines()]
    rows[0]["raw_reasoning"] = "private trajectory"
    table.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(release_audit.ReleaseAuditError, match="fields changed"):
        release_audit.audit_release(tmp_path, [*candidates, *result_paths])


def test_partial_or_unreviewed_public_results_path_fails_closed(tmp_path: Path) -> None:
    candidates = _safe_release_tree(tmp_path)
    extra = tmp_path / "reports/results/private_dump.json"
    extra.parent.mkdir(parents=True, exist_ok=True)
    extra.write_text("{}\n", encoding="utf-8")

    with pytest.raises(release_audit.ReleaseAuditError, match="explicit public-release allowlist"):
        release_audit.audit_release(
            tmp_path,
            [*candidates, "reports/results/private_dump.json"],
        )


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


def test_intermediate_symlink_component_is_rejected(tmp_path: Path) -> None:
    candidates = _safe_release_tree(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "artifact.md").write_text("external bytes\n", encoding="utf-8")
    linked = tmp_path / "docs/linked"
    linked.parent.mkdir(parents=True, exist_ok=True)
    linked.symlink_to(outside, target_is_directory=True)

    with pytest.raises(release_audit.ReleaseAuditError, match="symlinks"):
        release_audit.audit_release(tmp_path, [*candidates, "docs/linked/artifact.md"])


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


def test_primary_reasoning_manifest_is_never_public(tmp_path: Path) -> None:
    candidates = _safe_release_tree(tmp_path)
    relative = "data/manifests/anchor_manifest.json"
    path = tmp_path / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"reasoning": "private model trajectory"}) + "\n")

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
    relative = "data/manifests/upstream_reference.json"
    path = tmp_path / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    escaped_path = "\\u002fUsers\\u002falice\\u002fprivate.json"
    path.write_text('{"artifact":"' + escaped_path + '"}\n', encoding="utf-8")

    with pytest.raises(release_audit.ReleaseAuditError, match="macos_user_absolute_path"):
        release_audit.audit_release(tmp_path, [*candidates, relative])


def test_runpod_pod_and_gpu_identifiers_fail_closed(tmp_path: Path) -> None:
    candidates = _safe_release_tree(tmp_path)
    manifest_relative = "data/manifests/upstream_reference.json"
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


def test_literal_ssh_pod_and_session_prose_fail_but_placeholders_pass(
    tmp_path: Path,
) -> None:
    candidates = _safe_release_tree(tmp_path)
    relative = "docs/runtime-access.md"
    path = tmp_path / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    endpoint = "ssh " + "root@" + "203.0.113.10" + " -p " + "12345"
    path.write_text(endpoint + "\n", encoding="utf-8")
    with pytest.raises(release_audit.ReleaseAuditError, match="literal_ssh"):
        release_audit.audit_release(tmp_path, [*candidates, relative])

    pod_identifier = "p0d" + "9t4q2z8r6w3x1"
    path.write_text("RunPod Pod ID: " + pod_identifier + "\n", encoding="utf-8")
    with pytest.raises(release_audit.ReleaseAuditError, match="raw_runpod_pod_identifier"):
        release_audit.audit_release(tmp_path, [*candidates, relative])

    session_identifier = "a1" * 32
    path.write_text("GPU session ID: " + session_identifier + "\n", encoding="utf-8")
    with pytest.raises(release_audit.ReleaseAuditError, match="raw_gpu_session_identifier"):
        release_audit.audit_release(tmp_path, [*candidates, relative])

    path.write_text(
        "ssh root@<host> -p <port>\n"
        "RunPod Pod ID: <redacted>\n"
        "GPU session ID: ${GPU_BUDGET_SESSION_ID}\n"
        "GPU session ID: phase-session-opaque\n"
        "Direct SSH readiness and the Pod lifecycle are documented without identifiers.\n",
        encoding="utf-8",
    )
    assert release_audit.audit_release(tmp_path, [*candidates, relative])["status"] == "passed"


@pytest.mark.parametrize(
    "content, finding",
    [
        (
            "Connect with `ssh root@203.0.113.10 -p 12345` now.\n",
            "literal_ssh",
        ),
        ("- ssh root@203.0.113.10 -p 12345\n", "literal_ssh"),
        ("Use Pod `p0d9t4q2z8r6w3x1` for the run.\n", "raw_runpod_pod_identifier"),
        ("Use Pod p0d9t4q2z8r6w3x1 for the run.\n", "raw_runpod_pod_identifier"),
    ],
)
def test_markdown_inline_and_bulleted_infrastructure_prose_fails_closed(
    tmp_path: Path,
    content: str,
    finding: str,
) -> None:
    candidates = _safe_release_tree(tmp_path)
    relative = "docs/inline-runtime-access.md"
    path = tmp_path / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")

    with pytest.raises(release_audit.ReleaseAuditError, match=finding):
        release_audit.audit_release(tmp_path, [*candidates, relative])


@pytest.mark.parametrize(
    "prefix",
    [
        "Use GPU session `{value}` for the run.\n",
        "- Continue with session {value}.\n",
        "The session is `{value}`.\n",
        "session nonce: {value}\n",
    ],
)
def test_concrete_session_nonce_in_markdown_prose_fails_closed(
    tmp_path: Path, prefix: str
) -> None:
    candidates = _safe_release_tree(tmp_path)
    relative = "docs/session-runtime-access.md"
    path = tmp_path / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(prefix.format(value="a1" * 32), encoding="utf-8")

    with pytest.raises(release_audit.ReleaseAuditError, match="raw_gpu_session_identifier"):
        release_audit.audit_release(tmp_path, [*candidates, relative])


@pytest.mark.parametrize(
    "content",
    [
        "Use GPU session `phase-session-opaque` for the run.\n",
        "- The session lifecycle is authenticated by a content hash.\n",
        "GPU session ID: ${GPU_BUDGET_SESSION_ID}\n",
        "GPU session ID: session-placeholder\n",
    ],
)
def test_session_prose_placeholders_and_descriptions_remain_public(
    tmp_path: Path, content: str
) -> None:
    candidates = _safe_release_tree(tmp_path)
    relative = "docs/session-placeholder.md"
    path = tmp_path / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")

    assert release_audit.audit_release(
        tmp_path, [*candidates, relative]
    )["status"] == "passed"


def test_arbitrary_or_credential_bearing_figure_is_rejected(tmp_path: Path) -> None:
    candidates = _safe_release_tree(tmp_path)
    arbitrary = tmp_path / "reports/figures/private-trace.png"
    arbitrary.parent.mkdir(parents=True, exist_ok=True)
    arbitrary.write_bytes(b"not a reviewed figure")
    with pytest.raises(release_audit.ReleaseAuditError, match="explicit public-release allowlist"):
        release_audit.audit_release(
            tmp_path,
            [*candidates, "reports/figures/private-trace.png"],
        )

    credential = tmp_path / "reports/figures/smoke/first_vs_final_bias.png"
    credential.parent.mkdir(parents=True, exist_ok=True)
    credential.write_bytes(("sk-" + "proj-" + "A" * 32).encode())
    with pytest.raises(release_audit.ReleaseAuditError, match="possible secret"):
        release_audit.audit_release(
            tmp_path,
            [*candidates, "reports/figures/smoke/first_vs_final_bias.png"],
        )


def test_tampered_primary_figure_fails_trusted_regeneration(tmp_path: Path) -> None:
    candidates = _safe_release_tree(tmp_path)
    result_paths = _write_public_result_fixture(tmp_path)
    figure = tmp_path / "reports/figures/first_vs_final_bias.png"
    figure.write_bytes(figure.read_bytes() + b"tampered")

    with pytest.raises(release_audit.ReleaseAuditError, match="trusted regeneration"):
        release_audit.audit_release(tmp_path, [*candidates, *result_paths])


def test_unavailable_lens_rejects_stale_or_arbitrary_primary_heatmap(tmp_path: Path) -> None:
    candidates = _safe_release_tree(tmp_path)
    result_paths = _write_public_result_fixture(tmp_path)
    relative = "reports/figures/lens_layer_position_heatmap.png"
    figure = tmp_path / relative
    figure.write_bytes(b"arbitrary stale lens figure")

    with pytest.raises(release_audit.ReleaseAuditError, match="figure inventory"):
        release_audit.audit_release(tmp_path, [*candidates, *result_paths, relative])


@pytest.mark.parametrize(
    "private_field",
    ["raw_response", "response_body", "provider_response", "final_response", "choices"],
)
def test_private_provider_response_bodies_fail_closed(
    tmp_path: Path,
    private_field: str,
) -> None:
    candidates = _safe_release_tree(tmp_path)
    relative = "data/manifests/upstream_reference.json"
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
    relative = "data/manifests/upstream_reference.json"
    path = tmp_path / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    embedded = "OPENROUTER_" + "API_KEY=live-credential-value"
    path.write_text(json.dumps({"log": embedded}) + "\n", encoding="utf-8")

    with pytest.raises(release_audit.ReleaseAuditError, match="embedded_secret_or_credential"):
        release_audit.audit_release(tmp_path, [*candidates, relative])


def test_duplicate_structured_keys_cannot_bypass_content_scan(tmp_path: Path) -> None:
    candidates = _safe_release_tree(tmp_path)
    relative = "data/manifests/upstream_reference.json"
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


def test_staged_secret_is_audited_even_when_worktree_copy_is_safe(tmp_path: Path) -> None:
    candidates = _safe_release_tree(tmp_path)
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "add", "--", *candidates], check=True)
    readme = tmp_path / "README.md"
    staged_value = "sk-or-" + "v1-" + "A" * 32
    readme.write_text("staged credential: " + staged_value + "\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "--", "README.md"], check=True)
    readme.write_text("# safe working copy\n", encoding="utf-8")

    discovered = release_audit.discover_release_candidates(tmp_path)
    index_entries = release_audit.discover_git_index_entries(tmp_path)
    with pytest.raises(release_audit.ReleaseAuditError, match="Git index"):
        release_audit.audit_release(
            tmp_path,
            discovered,
            git_index_entries=index_entries,
        )


def test_current_git_release_candidates_pass() -> None:
    root = SCRIPT.parents[1]
    candidates = release_audit.discover_release_candidates(root)

    result = release_audit.audit_release(
        root,
        candidates,
        git_index_entries=release_audit.discover_git_index_entries(root),
    )

    assert result["status"] == "passed"
