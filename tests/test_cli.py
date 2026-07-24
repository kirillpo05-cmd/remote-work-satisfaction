"""Smoke tests for the CLI surface (SPEC M7). The heavy commands are exercised
against the real artifacts in the phase gates, not here."""

import pytest

from rwsat.cli import main


def test_validate_command_passes_on_the_committed_dataset(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["validate"]) == 0
    out = capsys.readouterr().out
    assert '"is_valid": true' in out


def test_unknown_command_is_rejected() -> None:
    with pytest.raises(SystemExit):
        main(["tune-hyperparameters"])
