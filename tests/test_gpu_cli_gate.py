from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import pytest

import model_forensics.cli as cli


def _config(tmp_path: Path) -> SimpleNamespace:
    source = tmp_path / "config" / "run.yaml"
    source.parent.mkdir(parents=True)
    source.write_text("schema_version: 1\n", encoding="utf-8")
    return SimpleNamespace(
        source_path=source,
        paths=SimpleNamespace(manifest_dir="data/manifests"),
    )


def _gate() -> SimpleNamespace:
    return SimpleNamespace(
        bindings=SimpleNamespace(caps_usd=SimpleNamespace(gpu=220.0, api=100.0, total=325.0))
    )


def test_gpu_only_parsers_expose_private_session_gate_without_secret_value() -> None:
    parser = cli.build_parser()
    for command, extra in (
        ("behavior-generate", ["--phase", "baseline"]),
        ("resample-generate", []),
        ("lens", []),
    ):
        parsed = parser.parse_args(
            [
                command,
                "--config",
                "config/run_122b.yaml",
                *extra,
                "--gpu-budget-reservation",
                ".runpod/reservations/phase.json",
                "--gpu-session-directory",
                ".runpod/sessions/abc",
            ]
        )
        assert parsed.gpu_session_id_env == "GPU_BUDGET_SESSION_ID"
        assert not hasattr(parsed, "gpu_session_id")


def test_in_process_gpu_gate_reads_nonce_only_from_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private = tmp_path / ".runpod"
    receipt = private / "reservations" / "phase.json"
    digest = "a" * 64
    session = private / "sessions" / digest
    receipt.parent.mkdir(parents=True)
    session.mkdir(parents=True)
    receipt.write_text("{}\n", encoding="utf-8")
    secret = "never-persist-this-session-nonce"
    monkeypatch.setenv("GPU_BUDGET_SESSION_ID", secret)
    monkeypatch.setattr(
        cli,
        "load_gpu_phase_budget_reservation",
        lambda path: SimpleNamespace(session_hash=f"sha256:{digest}"),
    )
    observed: dict[str, object] = {}

    def validate(**kwargs: object) -> dict[str, object]:
        observed.update(kwargs)
        return {
            "protocol_version": "active-runpod-session-v1",
            "phase": "resample_gpu",
            "passed": True,
            "record_hash": "sha256:gate",
        }

    monkeypatch.setattr(cli, "validate_active_runpod_session", validate)
    payload = cli._validate_active_gpu_session(
        argparse.Namespace(
            gpu_budget_reservation=str(receipt),
            gpu_session_directory=str(session),
            gpu_session_id_env="GPU_BUDGET_SESSION_ID",
            cost_ledger=None,
        ),
        config=_config(tmp_path),
        gate=_gate(),
        command_phase="resample_gpu",
    )

    assert observed["session_id"] == secret
    assert secret not in repr(payload)
    assert payload["passed"] is True


def test_in_process_gpu_gate_fails_before_receipt_load_when_runtime_args_absent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = False

    def forbidden_load(path: object) -> object:
        nonlocal loaded
        loaded = True
        raise AssertionError(path)

    monkeypatch.setattr(cli, "load_gpu_phase_budget_reservation", forbidden_load)
    with pytest.raises(cli.CLIError, match="requires --gpu-budget-reservation"):
        cli._validate_active_gpu_session(
            argparse.Namespace(
                gpu_budget_reservation=None,
                gpu_session_directory=None,
                gpu_session_id_env="GPU_BUDGET_SESSION_ID",
                cost_ledger=None,
            ),
            config=_config(tmp_path),
            gate=_gate(),
            command_phase="lens_gpu",
        )
    assert loaded is False
