from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _text(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


def test_release_docs_state_no_results_and_the_frozen_primary_boundaries() -> None:
    readme = _text("README.md")
    preregistration = _text("config/preregistration.yaml")

    assert "does **not** contain a completed primary run" in readme
    assert "blind external" in readme
    assert "unconditionally by 10 more" in readme
    assert "both blind external final-outcome adjudicators for every" in readme
    assert "Only exact known status-and-value consensus" in readme
    assert "validation-only mode" in readme
    assert "adaptive top-up" not in readme.lower()
    assert "stage_two_policy: unconditional_additional_10_per_anchor_arm" in preregistration
    assert "primary_measurement: blind_external_adjudication" in preregistration
    for excluded_category in (
        "replication",
        "sentence_resampling_generation",
        "model_and_lens_download",
        "document_production",
    ):
        assert f"  - {excluded_category}" in preregistration


def test_prereg_v3_freezes_lens_corroboration_and_release_authentication() -> None:
    readme = _text("README.md")
    preregistration = _text("config/preregistration.yaml")
    lens_contract = _text("docs/lens-position-input.md")

    for phrase in (
        "strictly_positive_J_and_R_tau_a",
        "p_value_and_leave_one_out_are_reported_descriptors_not_gate_requirements",
        "lens_compatibility_prefix_manifest.json",
        "lens_release_authorization.json",
        "pinned-tokenizer recomputation",
    ):
        assert phrase in preregistration or phrase in lens_contract or phrase in readme
    assert "exactly_4_above_plus_4_below_and_576_permutations" in preregistration
    assert "every smoke analysis row" in readme.lower()


def test_prereg_v3_freezes_exact_resampling_attrition_floors() -> None:
    preregistration = _text("config/preregistration.yaml")

    for phrase in (
        "resampling_generation_attrition_floors_v1",
        "minimum_overall_generation_valid_rate: 0.95",
        "expected_per_anchor_arm: 20",
        "minimum_anchor_arm_valid_count: 18",
        "minimum_anchor_pair_complete_count: 16",
        "maximum_anchor_arm_valid_rate_gap: 0.10",
        "exact_anchor_sample_index_stage_and_seed",
    ):
        assert phrase in preregistration


def test_runpod_docs_match_bootstrap_watchdog_and_budget_interfaces() -> None:
    runpod = _text("RUNPOD.md")
    bootstrap = _text("scripts/bootstrap_gpu.sh")

    assert 'if [[ "$#" -ne 22 ]]' in bootstrap
    for value in ("USD 220", "USD 100", "USD 5", "USD 325"):
        assert value in runpod
    for argument in (
        "--pod-id",
        "--expected-gpu-family",
        "--expected-gpu-count",
        "--maximum-approved-hourly-per-gpu-usd",
        "--gpu-hard-stop-usd",
        "--maximum-runtime-hours",
        "--prior-committed-gpu-usd",
        "--state",
    ):
        assert argument in bootstrap
    assert 'with_watchdog_credentials env PYTHONPATH="$PWD/src" nohup' in bootstrap
    assert "python3 scripts/runpod_watchdog.py" in bootstrap
    assert "v1 `costPerHr`" in runpod
    assert "`provider_evidence_unavailable`" in runpod
    assert "never turns absence into a false provider attestation" in runpod
    assert "`POST /v1/pods/{podId}/stop` with no request body" in runpod
    assert "config/gpu_lock.yaml" in runpod
    assert "container image digest" in runpod
    assert "wheel hash" in runpod
    assert "semantic-stack hash" in runpod
    assert "zero\nfabricated lens rows" in runpod
    assert "GPU_BUDGET_SESSION_ID" in runpod
    assert "scripts/gpu_budget_reserve.py" in runpod
    assert "scripts/runpod_active_session_verify.py" in runpod
    assert "scripts/gpu_budget_settle.py" in runpod
    assert "scripts/freeze_paid_bundle.py" in _text("README.md")
    assert ".runpod/sessions/" in runpod
    assert "stopped_confirmed" in runpod
    assert "settled" in runpod
    assert "data/manifests/runpod_watchdog.json" not in runpod
    assert "data/manifests/gpu_preflight.json" not in runpod
    assert "schema-v2" in runpod
    assert "full\nsoftware/GPU lock" in runpod
    assert "make behavior-baseline-generate" in runpod
    assert "make behavior-baseline-adjudicate" in runpod
    assert "make behavior-treatment-generate" in runpod
    assert "make behavior-treatment-adjudicate" in runpod
    assert "make resample-generate" in runpod
    assert "make resample-adjudicate" in runpod
    assert "always validation-only aliases" in runpod
    assert "When its\ncanonical JSONL is absent, `make resample`" not in runpod
    assert "PER_GPU_RATE=" not in runpod
    assert "--quote-hourly-per-gpu-usd" not in runpod


def test_make_dry_runs_expose_the_real_nonsecret_cli_arguments() -> None:
    common = ["make", "-n", "-C", str(ROOT)]
    sample = subprocess.run(
        [*common, "sample"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert "model_forensics.cli sample" in sample
    assert "--output" in sample
    assert "--sampling-manifest" in sample
    assert "--paid-approval" not in sample
    assert "--judge-model" not in sample
    assert "API_KEY" not in sample

    behavior_gpu = subprocess.run(
        [*common, "behavior-baseline-generate"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    for argument in (
        "model_forensics.cli behavior-generate",
        "--phase baseline",
        "--gpu-lock",
        "--gpu-quote-lock",
        "--api-quote-lock",
        "--paid-approval",
        "--paid-receipt-dir",
        "--gpu-budget-reservation",
        "--gpu-session-directory",
        "--gpu-session-id-env",
        "--cost-ledger",
    ):
        assert argument in behavior_gpu
    assert "--judge-model" not in behavior_gpu
    assert "API_KEY" not in behavior_gpu

    behavior_api = subprocess.run(
        [*common, "behavior-treatment-adjudicate"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert "model_forensics.cli behavior-adjudicate" in behavior_api
    assert "--phase treatment" in behavior_api
    assert "--baseline-adjudication-checkpoint-dir" in behavior_api
    assert "--paid-approval" in behavior_api
    assert "--gpu-session-directory" not in behavior_api
    assert "--judge-model" not in behavior_api

    anchors = subprocess.run(
        [*common, "anchors"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert "--paid-approval" in anchors
    assert "--classifier-a-model" not in anchors
    assert "--classifier-b-model" not in anchors

    resample_gpu = subprocess.run(
        [*common, "resample-generate"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    for argument in (
        "--sampling-manifest",
        "--checkpoint-dir",
        "--gpu-budget-reservation",
        "--gpu-session-directory",
        "--cost-ledger",
    ):
        assert argument in resample_gpu
    assert "--judge-model" not in resample_gpu

    resample_api = subprocess.run(
        [*common, "resample-adjudicate"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert "--generation-checkpoint-dir" in resample_api
    assert "--paid-approval" in resample_api
    assert "--judge-model" not in resample_api
    assert "--classifier-a-model" not in resample_api

    resample_validation = subprocess.run(
        [*common, "resample"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert "model_forensics.cli resample" in resample_validation
    assert "--input" in resample_validation
    assert "--paid-approval" not in resample_validation

    reproduce_results = subprocess.run(
        [*common, "reproduce-results"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert "scripts/reproduce_results.py" in reproduce_results
    assert "--results-dir" in reproduce_results
    assert "--figure-dir" in reproduce_results
    assert "model_forensics.cli analyze" not in reproduce_results
    assert "data/raw" not in reproduce_results
    assert "API_KEY" not in reproduce_results

    reserve = subprocess.run(
        [*common, "gpu-reserve", "GPU_PHASE=behavior_baseline_gpu"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert "scripts/gpu_budget_reserve.py" in reserve
    assert "load_gpu_quote_lock" in reserve
    assert "--approved-phase-runtime-hours" in reserve
    assert "--quote-hourly-per-gpu-usd" in reserve
    assert "--running-storage-hourly-usd" in reserve

    bootstrap_target = subprocess.run(
        [*common, "gpu-bootstrap", "GPU_PHASE=behavior_baseline_gpu"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert "scripts/extract_gpu_bootstrap_inputs.py" in bootstrap_target
    assert "scripts/bootstrap_gpu.sh" in bootstrap_target
    assert "provider_gpu_id" in bootstrap_target
    assert "allowed_cuda_versions" in bootstrap_target
    assert "data_center_ids" in bootstrap_target
    assert "storage_rate" in bootstrap_target

    active_verify = subprocess.run(
        [*common, "gpu-active-verify", "GPU_PHASE=resample_gpu"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert "scripts/runpod_active_session_verify.py" in active_verify
    assert "--session-directory" in active_verify
    assert "--reservation-receipt" in active_verify

    settle = subprocess.run(
        [
            *common,
            "gpu-settle",
            "GPU_PHASE=lens_gpu",
            "PROVIDER_INCURRED_USD=1.25",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert "scripts/gpu_budget_settle.py" in settle
    assert "--watchdog-state" in settle
    assert "--provider-incurred-usd" in settle
    assert "API_KEY" not in "\n".join(
        (sample, behavior_gpu, behavior_api, anchors, resample_gpu, resample_api)
    )
    assert (
        "scripts/investigation_timer.py status"
        in subprocess.run(
            [*common, "time-status"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )


def test_makefile_scopes_provider_credentials_to_paid_targets() -> None:
    makefile = _text("Makefile")

    assert re.search(
        r"(?m)^export HF_TOKEN OPENROUTER_API_KEY RUNPOD_API_KEY$", makefile
    ) is None
    assert "unexport HF_TOKEN OPENROUTER_API_KEY RUNPOD_API_KEY" in makefile
    assert "$(HF_TARGETS): export HF_TOKEN := $(LOCAL_HF)" in makefile
    assert (
        "$(OPENROUTER_TARGETS): export OPENROUTER_API_KEY := $(LOCAL_OPENROUTER)"
        in makefile
    )
    assert "gpu-bootstrap: export RUNPOD_API_KEY := $(LOCAL_RUNPOD)" in makefile
    release_recipe = makefile.split("release-check:", 1)[1]
    assert "HF_TOKEN" not in release_recipe
    assert "OPENROUTER_API_KEY" not in release_recipe
    assert "RUNPOD_API_KEY" not in release_recipe

def test_markdown_fences_and_relative_links_are_valid() -> None:
    for name in ("README.md", "RUNPOD.md", "THIRD_PARTY_NOTICES.md"):
        text = _text(name)
        assert text.count("```") % 2 == 0, f"unbalanced code fence in {name}"
        for target in re.findall(r"\[[^]]+\]\(([^)]+)\)", text):
            if "://" in target or target.startswith(("#", "mailto:")):
                continue
            clean = target.split("#", 1)[0]
            assert (ROOT / clean).exists(), f"broken relative link in {name}: {target}"


def test_third_party_notice_preserves_no_license_nonredistribution_boundary() -> None:
    notice = _text("THIRD_PARTY_NOTICES.md")

    assert "no license file was present" in notice
    assert "no source files, prompt files, figures, or raw traces" in notice
    assert "Absence of a license is not permission to redistribute" in notice
    assert "does not relicense" in notice
