"""End-to-end smoke test for mflux-save-advanced CLI surface.

This is a fast smoke test (no HuggingFace downloads, no real model loading).
It exercises:
  - the mflux-save-advanced console script entry point
  - argument parsing
  - source-format detection (the bf16 pytorch guard)
  - the mutually-exclusive flag guard
  - the file-format error path (what the user sees when they point at junk)

The full end-to-end (loading a real model and saving a mixed-precision copy)
is exercised by tests/model_saving/* with @pytest.mark.slow. We don't
duplicate that here.
"""
import json
import subprocess
import sys
from pathlib import Path

import mlx.core as mx
import pytest


CONSOLE = "/Users/ianscrivener/zCode_26-6/mflux/.venv-313/bin/mflux-save-advanced"


def _write_bf16_pytorch_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    weights = {"linear.weight": mx.zeros((4, 4), dtype=mx.bfloat16)}
    mx.save_safetensors(str(path / "model.safetensors"), weights, {"format": "pt"})
    (path / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"format": "pt"},
        "weight_map": {"linear.weight": "model.safetensors"},
    }))
    return path


def test_help_runs():
    """`mflux-save-advanced --help` prints the help and exits 0."""
    result = subprocess.run(
        [CONSOLE, "--help"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0
    assert "--bf16_model" in result.stdout
    assert "--nvfp4_model" in result.stdout
    assert "--quantize_vae" in result.stdout
    assert "--quantize_transformer" in result.stdout
    assert "--quantize_text_encoder" in result.stdout
    assert "--path" in result.stdout
    assert "--force" in result.stdout


def test_rejects_unknown_source_dir(tmp_path: Path):
    """Pointing at a non-existent path produces a clean format error."""
    result = subprocess.run(
        [
            CONSOLE,
            "--bf16_model", str(tmp_path / "does_not_exist"),
            "--path", str(tmp_path / "out"),
            "--force",
        ],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode != 0
    assert "not a bf16 pytorch-safetensors model" in (result.stderr + result.stdout)


def test_rejects_empty_dir_as_source(tmp_path: Path):
    """An empty directory is not a bf16 pytorch-safetensors model — clear error."""
    empty = tmp_path / "empty"
    empty.mkdir()
    result = subprocess.run(
        [
            CONSOLE,
            "--bf16_model", str(empty),
            "--path", str(tmp_path / "out"),
            "--force",
        ],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode != 0
    assert "not a bf16 pytorch-safetensors model" in (result.stderr + result.stdout)


def test_rejects_mflux_saved_bf16_as_source(tmp_path: Path):
    """An mflux-saved bf16 model is rejected by Task 1 (must be pytorch source)."""
    saved = tmp_path / "saved"
    saved.mkdir()
    weights = {"linear.weight": mx.zeros((4, 4), dtype=mx.bfloat16)}
    # mflux-saved format HAS mflux_version metadata.
    mx.save_safetensors(
        str(saved / "model.safetensors"), weights,
        {"format": "pt", "mflux_version": "0.18.0", "quantization_level": "bf16"},
    )
    (saved / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"format": "pt", "mflux_version": "0.18.0", "quantization_level": "bf16"},
        "weight_map": {"linear.weight": "model.safetensors"},
    }))
    result = subprocess.run(
        [
            CONSOLE,
            "--bf16_model", str(saved),
            "--path", str(tmp_path / "out"),
            "--force",
        ],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode != 0
    assert "not a bf16 pytorch-safetensors model" in (result.stderr + result.stdout)


def test_rejects_both_model_flags(tmp_path: Path):
    """Passing both --bf16_model and --nvfp4_model is a CLI error."""
    src = tmp_path / "src"
    src.mkdir()
    result = subprocess.run(
        [
            CONSOLE,
            "--bf16_model", str(src),
            "--nvfp4_model", str(src),
            "--path", str(tmp_path / "out"),
            "--force",
        ],
        capture_output=True, text=True, timeout=30,
    )
    # argparse's mutually_exclusive_group exits with code 2
    assert result.returncode == 2
