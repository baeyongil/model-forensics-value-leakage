from __future__ import annotations

import threading
from pathlib import Path

import pytest

from model_forensics import cli
from model_forensics.paid_bundle_rotation import (
    PaidBundleRotationError,
    paid_bundle_lock,
)


def test_active_api_command_excludes_paid_bundle_rotation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "config" / "run_122b.yaml"
    config.parent.mkdir()
    config.write_text("test fixture\n", encoding="utf-8")
    (tmp_path / ".runpod").mkdir(mode=0o700)
    entered = threading.Event()
    release = threading.Event()
    failures: list[BaseException] = []

    def blocking_api_handler(_args: object) -> dict[str, bool]:
        entered.set()
        if not release.wait(timeout=5):
            raise AssertionError("test did not release the API command")
        return {"passed": True}

    monkeypatch.setattr(cli, "_command_behavior_adjudicate", blocking_api_handler)

    def invoke() -> None:
        try:
            cli.main(
                [
                    "behavior-adjudicate",
                    "--config",
                    str(config),
                    "--phase",
                    "baseline",
                ]
            )
        except BaseException as exc:  # pragma: no cover - surfaced below
            failures.append(exc)

    thread = threading.Thread(target=invoke)
    thread.start()
    assert entered.wait(timeout=5)
    try:
        with pytest.raises(PaidBundleRotationError, match="already held"):
            with paid_bundle_lock(project_root=tmp_path, exclusive=True):
                pass
    finally:
        release.set()
        thread.join(timeout=5)

    assert not thread.is_alive()
    assert failures == []
