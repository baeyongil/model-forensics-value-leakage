from __future__ import annotations

import argparse
from pathlib import Path

import pytest

import model_forensics.cli as cli

ROOT = Path(__file__).resolve().parents[1]
PRIMARY_CONFIG = ROOT / "config" / "run_122b.yaml"


@pytest.mark.parametrize("command", ("sample", "resample"))
def test_validation_aliases_reject_legacy_freeform_paid_routes(
    command: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli.build_parser().parse_args(
            [command, "--config", str(PRIMARY_CONFIG), "--judge-model", "provider/model"]
        )
    assert exc_info.value.code == 2
    assert "unrecognized arguments: --judge-model" in capsys.readouterr().err


def test_resample_validation_alias_never_falls_back_to_paid_execution(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli.main(
            [
                "resample",
                "--config",
                str(PRIMARY_CONFIG),
                "--input",
                str(tmp_path / "absent.jsonl"),
            ]
        )
    assert exc_info.value.code == 2
    message = capsys.readouterr().err
    assert "validation-only" in message
    assert "resample-generate followed by resample-adjudicate" in message


def test_resample_adjudicate_has_no_model_or_price_override_surface() -> None:
    parser = cli.build_parser()
    action = next(item for item in parser._actions if isinstance(item, argparse._SubParsersAction))
    adjudicate = action.choices["resample-adjudicate"]
    option_strings = {option for item in adjudicate._actions for option in item.option_strings}
    assert "--paid-approval" in option_strings
    assert "--api-quote-lock" in option_strings
    assert "--judge-model" not in option_strings
    assert "--classifier-a-model" not in option_strings
    assert "--cost-ledger" not in option_strings


def test_resample_adjudicate_fails_at_approval_before_any_model_or_client(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    constructed: list[str] = []

    def forbidden(*_args: object, **_kwargs: object) -> object:
        constructed.append("constructed")
        raise AssertionError("approval failure must precede every model/client")

    monkeypatch.setattr(cli, "OpenRouterJSONClient", forbidden)
    monkeypatch.setattr(cli, "OpenRouterAdjudicationCaller", forbidden)
    monkeypatch.setattr(cli, "PinnedSentenceTransformerEmbedder", forbidden)
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["resample-adjudicate", "--config", str(PRIMARY_CONFIG)])
    assert exc_info.value.code == 2
    assert constructed == []
    assert "quote lock" in capsys.readouterr().err.lower()
