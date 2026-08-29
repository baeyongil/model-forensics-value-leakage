from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "extract_gpu_bootstrap_inputs.py"
SPEC = importlib.util.spec_from_file_location("extract_gpu_bootstrap_inputs", SCRIPT)
assert SPEC and SPEC.loader
extractor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(extractor)

OFFICIAL_CATALOG_URL = (
    "https://api.runpod.io/v2/catalog/gpus"
    "?include=AVAILABILITY&product=POD&count=8&cloud=SECURE"
)


def _quote(source_url: str) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": 1,
        "provider": "runpod",
        "quote_id": "runpod-h100-reviewed-20260829-001",
        "gpu_family": "H100_80GB",
        "provider_gpu_id": "NVIDIA H100 80GB HBM3",
        "cloud_type": "SECURE",
        "allowed_cuda_versions": ["12.8"],
        "data_center_ids": ["US-IL-1"],
        "gpu_count": 8,
        "container_disk_gb": 50,
        "volume_disk_gb": 650,
        "usd_per_gpu_hour": 3.29,
        "running_storage_usd_per_hour": 700 * 0.10 / 720,
        "quoted_at": "2026-08-29T18:20:38Z",
        "phase_runtime_allocations": [
            {"command_phase": phase, "maximum_runtime_hours": 1.0}
            for phase in extractor.GPU_PHASES
        ],
        "source_url": source_url,
    }
    payload["content_hash"] = extractor._stable_hash(payload)
    return payload


def test_quote_loader_accepts_official_catalog_deterministic_query(tmp_path: Path) -> None:
    path = tmp_path / "gpu_quote_lock.json"
    path.write_text(json.dumps(_quote(OFFICIAL_CATALOG_URL)), encoding="utf-8")

    loaded = extractor.load_quote(path, phase="behavior_baseline_gpu")

    assert loaded["source_url"] == OFFICIAL_CATALOG_URL


@pytest.mark.parametrize(
    "url",
    [
        "https://user@example.com/v2/catalog/gpus?count=8",
        "https://api.runpod.io/v2/catalog/gpus?count=8#availability",
        "https://api.runpod.io/v2/catalog/gpus?count=8%0Acloud=SECURE",
        "https://api.runpod.io/v2/catalog/gpus?count=8;cloud=SECURE",
        "https://api.runpod.io/v2/catalog/gpus?count=8&count=4",
    ],
)
def test_quote_source_rejects_credentials_fragments_and_unsafe_queries(url: str) -> None:
    with pytest.raises(extractor.BootstrapInputError):
        extractor._credential_free_https(
            url,
            field="source_url",
            allow_deterministic_query=True,
        )


def test_exact_wheel_url_remains_query_free() -> None:
    with pytest.raises(extractor.BootstrapInputError):
        extractor._credential_free_https(
            "https://files.pythonhosted.org/vllm.whl?download=1",
            field="vLLM wheel URL",
        )
