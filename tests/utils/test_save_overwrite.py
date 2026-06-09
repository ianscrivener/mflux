"""Tests for the overwrite-confirmation helper used by mflux-save-advanced."""
from pathlib import Path
from unittest.mock import patch

import pytest

from mflux.utils.save_overwrite import confirm_overwrite


class _TtyStdin:
    """A stdin stand-in whose isatty() returns True."""

    def isatty(self) -> bool:
        return True


class _NonTtyStdin:
    """A stdin stand-in whose isatty() returns False."""

    def isatty(self) -> bool:
        return False


def test_force_skips_prompt(tmp_path: Path):
    target = tmp_path / "model"
    target.mkdir()
    # No patching of input — if force=True the function must return before touching stdin.
    confirm_overwrite(target, force=True)


def test_no_prompt_when_target_missing(tmp_path: Path):
    target = tmp_path / "missing"
    confirm_overwrite(target, force=False)


def test_proceeds_on_yes_when_target_exists(tmp_path: Path):
    target = tmp_path / "model"
    target.mkdir()
    with patch("sys.stdin", _TtyStdin()):
        with patch("builtins.input", return_value="y") as m_input:
            confirm_overwrite(target, force=False)
            assert m_input.called


def test_aborts_on_no_when_target_exists(tmp_path: Path):
    target = tmp_path / "model"
    target.mkdir()
    with patch("sys.stdin", _TtyStdin()):
        with patch("builtins.input", return_value="n"):
            with pytest.raises(SystemExit) as exc:
                confirm_overwrite(target, force=False)
            assert exc.value.code != 0


def test_aborts_on_non_tty(tmp_path: Path):
    target = tmp_path / "model"
    target.mkdir()
    with patch("sys.stdin", _NonTtyStdin()):
        with pytest.raises(SystemExit):
            confirm_overwrite(target, force=False)


def test_treats_yes_with_whitespace_as_yes(tmp_path: Path):
    target = tmp_path / "model"
    target.mkdir()
    with patch("sys.stdin", _TtyStdin()):
        with patch("builtins.input", return_value="  y  "):
            confirm_overwrite(target, force=False)  # should not raise
