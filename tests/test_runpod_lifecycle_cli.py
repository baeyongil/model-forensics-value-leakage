from __future__ import annotations

import pytest

from scripts import runpod_pod_lifecycle


@pytest.mark.parametrize("operation", ["create", "recover-create", "rearm"])
def test_paid_lifecycle_commands_default_to_canonical_cost_ledger(operation: str) -> None:
    argv = [
        operation,
        "--phase",
        "behavior_baseline_gpu",
        "--reservation",
        ".runpod/reservations/behavior_baseline_gpu.json",
    ]
    if operation == "rearm":
        argv.extend(
            [
                "--host-rearm-ack",
                ".runpod/sessions/test/host_rearm_watchdog_ack.json",
            ]
        )
    args = runpod_pod_lifecycle._parser().parse_args(argv)

    assert args.cost_ledger == "data/manifests/cost_ledger.yaml"


@pytest.mark.parametrize("operation", ["create", "recover-create"])
def test_frozen_cli_rejects_fresh_creation_before_provider_or_environment_access(
    operation: str,
) -> None:
    provider_called = False

    def forbidden_transport(**_kwargs: object) -> object:
        nonlocal provider_called
        provider_called = True
        raise AssertionError("provider transport must remain unreachable")

    arguments = [
        "--project-root",
        ".",
        operation,
        "--phase",
        "behavior_baseline_gpu",
        "--reservation",
        ".runpod/reservations/behavior_baseline_gpu.json",
    ]
    with pytest.raises(SystemExit, match="fresh Pod creation is disabled"):
        runpod_pod_lifecycle.main(arguments, transport=forbidden_transport)  # type: ignore[arg-type]
    assert provider_called is False
