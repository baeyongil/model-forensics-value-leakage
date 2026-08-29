from __future__ import annotations

import os
from pathlib import Path

import pytest

from model_forensics.qwen4b_smoke import run_qwen4b_prefix_smoke


@pytest.mark.integration
@pytest.mark.gpu
@pytest.mark.skipif(
    os.environ.get("RUN_QWEN4B_GPU_INTEGRATION") != "1",
    reason="set RUN_QWEN4B_GPU_INTEGRATION=1 only on the prepaid CUDA host",
)
def test_bounded_real_qwen4b_integration_gate(tmp_path: Path) -> None:
    """Opt-in real-model gate; normal local tests never load or download a model."""

    manifest = run_qwen4b_prefix_smoke(
        tmp_path / "qwen4b_integration_smoke.json",
        tensor_parallel_size=1,
        max_model_len=4096,
        rollout_max_tokens=1024,
        continuation_max_tokens=256,
    )

    assert manifest["status"] == "passed"
    assert manifest["experimental_sample"] is False
    assert manifest["primary_eligible"] is False
    assert manifest["paid_api_calls"] == 0
    assert set(manifest["raw_prefix_continuations"]) == {"retain", "resample"}
    assert manifest["lens_probe_grid"]["probe_cell_count"] == 15
    boundary = manifest["lens_probe_grid"]["transport_boundary"]
    assert boundary["activation_transport_executed"] is False
    assert boundary["fabricated_lens_record_count"] == 0
    assert manifest["analysis_evidence_handoff"]["analysis_ingest_allowed"] is False
