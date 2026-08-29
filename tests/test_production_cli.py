from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

from model_forensics import cli
from model_forensics.io import stable_hash, write_json
from model_forensics.sampling import FakeBackend

ROOT = Path(__file__).resolve().parents[1]


class FakePinnedBackend(FakeBackend):
    @property
    def provenance(self):  # type: ignore[no-untyped-def]
        return {
            "backend": "vllm_offline",
            "model_id": "Qwen/Qwen3.5-122B-A10B",
            "model_revision": "dc4d348443bc740c68e2d77492492c11606384d5",
            "revision": "dc4d348443bc740c68e2d77492492c11606384d5",
            "tokenizer_id": "Qwen/Qwen3.5-122B-A10B",
            "tokenizer_revision": "dc4d348443bc740c68e2d77492492c11606384d5",
            "dtype": "bfloat16",
            "tensor_parallel_size": 8,
            "max_model_len": 65_536,
            "chat_template_kwargs_hash": stable_hash({"enable_thinking": True}),
            "detokenization_kwargs_hash": stable_hash(
                {"skip_special_tokens": True, "spaces_between_special_tokens": True}
            ),
            "chat_template_hash": stable_hash({"chat_template": "frozen-test-template"}),
            "vllm_version": "0.28.0",
            "transformers_version": "5.5.3",
        }


def _gpu_binding() -> SimpleNamespace:
    return SimpleNamespace(
        count=8,
        family="H100_80GB",
        container_image_digest="vllm/vllm-openai@sha256:" + "1" * 64,
        vllm_wheel_sha256="2" * 64,
    )


def _args(tmp_path: Path, *, phase: str = "baseline") -> Namespace:
    return Namespace(
        config=str(ROOT / "config/run_122b.yaml"),
        phase=phase,
        checkpoint_dir=str(tmp_path / phase),
        thresholds=None,
        batch_size=16,
        max_new_batches=None,
        gpu_lock=None,
        gpu_quote_lock=None,
        paid_approval=None,
    )


def test_behavior_generate_validates_approval_before_backend_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []

    def reject(*args, **kwargs):  # type: ignore[no-untyped-def]
        events.append("approval")
        raise cli.CLIError("not approved")

    def forbidden_backend(*args, **kwargs):  # type: ignore[no-untyped-def]
        events.append("backend")
        raise AssertionError("backend must not be constructed")

    monkeypatch.setattr(cli, "_validate_paid_phase", reject)
    monkeypatch.setattr(cli, "VLLMOfflineBackend", forbidden_backend)
    with pytest.raises(cli.CLIError, match="not approved"):
        cli._command_behavior_generate(_args(tmp_path))
    assert events == ["approval"]


def test_behavior_generate_is_gpu_only_and_resume_skips_model_reload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []

    def approve(*args, **kwargs):  # type: ignore[no-untyped-def]
        events.append("approval")
        return SimpleNamespace(bindings=SimpleNamespace(gpu=_gpu_binding()))

    def backend(*args, **kwargs):  # type: ignore[no-untyped-def]
        events.append("backend")
        return FakePinnedBackend()

    monkeypatch.setattr(cli, "_validate_paid_phase", approve)
    monkeypatch.setattr(cli, "_authorize_paid_plan", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        cli,
        "_validate_active_gpu_session",
        lambda *args, **kwargs: {"record_hash": "sha256:" + "3" * 64},
    )
    monkeypatch.setattr(cli, "VLLMOfflineBackend", backend)
    result = cli._command_behavior_generate(_args(tmp_path))
    assert result["status"] == "complete"
    assert result["api_calls_performed"] == 0
    assert result["row_count"] == 70
    assert events == ["approval", "backend"]

    events.clear()
    resumed = cli._command_behavior_generate(_args(tmp_path))
    assert resumed["status"] == "complete"
    assert events == ["approval"]


def test_treatment_generation_requires_authenticated_thresholds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        cli,
        "_validate_paid_phase",
        lambda *args, **kwargs: SimpleNamespace(bindings=SimpleNamespace(gpu=_gpu_binding())),
    )
    monkeypatch.setattr(cli, "_authorize_paid_plan", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        cli,
        "_validate_active_gpu_session",
        lambda *args, **kwargs: {"record_hash": "sha256:" + "3" * 64},
    )
    args = _args(tmp_path, phase="treatment")
    with pytest.raises(cli.CLIError, match="thresholds are absent"):
        cli._command_behavior_generate(args)

    payload = {
        "schema_version": 1,
        "thresholds": {"giraffe": 41_000_000.0, "chicago_coffee": 1_000_000.0},
    }
    payload["manifest_hash"] = stable_hash(payload)
    threshold_path = tmp_path / "thresholds.json"
    write_json(threshold_path, payload)
    args.thresholds = str(threshold_path)
    monkeypatch.setattr(cli, "VLLMOfflineBackend", lambda *args, **kwargs: FakePinnedBackend())
    result = cli._command_behavior_generate(args)
    assert result["row_count"] == 240


def test_resample_generate_validates_approval_before_raw_prefix_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []

    def reject(*args, **kwargs):  # type: ignore[no-untyped-def]
        events.append("approval")
        raise cli.CLIError("not approved")

    def forbidden_backend(*args, **kwargs):  # type: ignore[no-untyped-def]
        events.append("backend")
        raise AssertionError("raw-prefix backend must not be constructed")

    monkeypatch.setattr(cli, "_validate_paid_phase", reject)
    monkeypatch.setattr(cli, "VLLMRawPrefixBackend", forbidden_backend)
    args = Namespace(
        config=str(ROOT / "config/run_122b.yaml"),
        rollouts=None,
        anchors=None,
        sampling_manifest=None,
        checkpoint_dir=str(tmp_path),
        microbatch_size=8,
        gpu_lock=None,
        gpu_quote_lock=None,
        paid_approval=None,
    )
    with pytest.raises(cli.CLIError, match="not approved"):
        cli._command_resample_generate(args)
    assert events == ["approval"]
