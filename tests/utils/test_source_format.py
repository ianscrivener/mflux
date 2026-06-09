"""Tests for the source-format detection helpers used by mflux-save-advanced."""
import json
from pathlib import Path
from unittest.mock import patch

import mlx.core as mx
import pytest

from mflux.utils.source_format import (
    detect_bf16_pytorch_safetensors,
    detect_nvfp4_pytorch_safetensors,
)


def _write_bf16_pytorch_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    weights = {"linear1.weight": mx.zeros((2, 2), dtype=mx.bfloat16)}
    # pytorch format does NOT include the mflux_version metadata key.
    mx.save_safetensors(str(path / "model.safetensors"), weights, {"format": "pt"})
    index = {
        "metadata": {"format": "pt"},
        "weight_map": {"linear1.weight": "model.safetensors"},
    }
    (path / "model.safetensors.index.json").write_text(json.dumps(index))
    return path


def _write_bf16_mflux_saved_dir(path: Path) -> Path:
    """An mflux-saved bf16 model — must NOT be detected as pytorch."""
    path.mkdir(parents=True, exist_ok=True)
    weights = {"linear1.weight": mx.zeros((2, 2), dtype=mx.bfloat16)}
    # mflux-saved format DOES include mflux_version metadata.
    mx.save_safetensors(
        str(path / "model.safetensors"),
        weights,
        {"format": "pt", "mflux_version": "0.18.0", "quantization_level": "bf16"},
    )
    (path / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"format": "pt", "quantization_level": "bf16", "mflux_version": "0.18.0"},
        "weight_map": {"linear1.weight": "model.safetensors"},
    }))
    return path


def _write_nvfp4_pytorch_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    weights = {"transformer.linear1.weight": mx.zeros((2, 2), dtype=mx.uint32)}
    # nvfp4 source: quantization_level=4, no mflux_version.
    mx.save_safetensors(str(path / "model.safetensors"), weights, {"quantization_level": "4", "format": "pt"})
    (path / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"quantization_level": "4", "format": "pt"},
        "weight_map": {"transformer.linear1.weight": "model.safetensors"},
    }))
    return path


def test_detects_bf16_pytorch_dir(tmp_path: Path):
    _write_bf16_pytorch_dir(tmp_path)
    assert detect_bf16_pytorch_safetensors(str(tmp_path)) is True


def test_rejects_empty_dir(tmp_path: Path):
    assert detect_bf16_pytorch_safetensors(str(tmp_path)) is False


def test_rejects_mflux_saved_bf16(tmp_path: Path):
    """An mflux-saved model is NOT a pytorch source — must be rejected by Task 1."""
    _write_bf16_mflux_saved_dir(tmp_path)
    assert detect_bf16_pytorch_safetensors(str(tmp_path)) is False


def test_detects_nvfp4_pytorch_dir(tmp_path: Path):
    _write_nvfp4_pytorch_dir(tmp_path)
    assert detect_nvfp4_pytorch_safetensors(str(tmp_path)) is True


def test_rejects_bf16_for_nvfp4_detector(tmp_path: Path):
    _write_bf16_pytorch_dir(tmp_path)
    assert detect_nvfp4_pytorch_safetensors(str(tmp_path)) is False
