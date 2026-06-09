"""Tests for per-component bits in ModelSaver."""
from pathlib import Path
from unittest.mock import MagicMock

import mlx.core as mx
from mlx import nn

from mflux.models.common.weights.saving.model_saver import ModelSaver


def _tiny_module() -> nn.Module:
    """A minimal nn.Module with one Linear that tree_flatten can walk."""
    m = nn.Linear(2, 2)
    return m


def _make_wrapper() -> object:
    """Wrap a tiny module as a `vae` attribute on a dummy namespace."""
    wrapper = type("W", (), {})()
    wrapper.vae = _tiny_module()
    return wrapper


def _read_shard_metadata(shard_path: Path) -> dict:
    """Read the safetensors metadata block from a single shard."""
    data = mx.load(str(shard_path), return_metadata=True)
    if not data or len(data) < 2:
        return {}
    return dict(data[1])


def test_int_bits_still_works(tmp_path: Path) -> None:
    """Backward compat: passing a plain int writes that int as metadata."""
    weight_def = MagicMock()
    weight_def.get_tokenizers.return_value = []
    comp = MagicMock()
    comp.model_attr = "vae"
    comp.hf_subdir = "vae"
    comp.weight_subkey = None
    weight_def.get_components.return_value = [comp]

    ModelSaver.save_model(
        model=_make_wrapper(),
        bits=4,
        base_path=str(tmp_path),
        weight_definition=weight_def,
    )
    shard = next((tmp_path / "vae").glob("*.safetensors"))
    md = _read_shard_metadata(shard)
    assert str(md.get("quantization_level")) == "4"


def test_dict_bits_writes_per_component_metadata(tmp_path: Path) -> None:
    """Passing a dict writes per-subdir metadata."""
    weight_def = MagicMock()
    weight_def.get_tokenizers.return_value = []
    comp = MagicMock()
    comp.model_attr = "vae"
    comp.hf_subdir = "vae"
    comp.weight_subkey = None
    weight_def.get_components.return_value = [comp]

    ModelSaver.save_model(
        model=_make_wrapper(),
        bits={"vae": 8, "transformer": 4},  # transformer is absent in this model
        base_path=str(tmp_path),
        weight_definition=weight_def,
    )
    shard = next((tmp_path / "vae").glob("*.safetensors"))
    md = _read_shard_metadata(shard)
    assert str(md.get("quantization_level")) == "8"


def test_dict_bits_with_bf16_writes_bf16_metadata(tmp_path: Path) -> None:
    """A dict value of None (the bf16 sentinel) writes 'bf16' as the metadata."""
    weight_def = MagicMock()
    weight_def.get_tokenizers.return_value = []
    comp = MagicMock()
    comp.model_attr = "vae"
    comp.hf_subdir = "vae"
    comp.weight_subkey = None
    weight_def.get_components.return_value = [comp]

    ModelSaver.save_model(
        model=_make_wrapper(),
        bits={"vae": None},
        base_path=str(tmp_path),
        weight_definition=weight_def,
    )
    shard = next((tmp_path / "vae").glob("*.safetensors"))
    md = _read_shard_metadata(shard)
    assert str(md.get("quantization_level")) == "bf16"


def test_dict_missing_subdir_falls_back_to_bf16(tmp_path: Path) -> None:
    """If a subdir is not in the dict, fall back to bf16 (preserve zero-config)."""
    weight_def = MagicMock()
    weight_def.get_tokenizers.return_value = []
    comp = MagicMock()
    comp.model_attr = "vae"
    comp.hf_subdir = "vae"
    comp.weight_subkey = None
    weight_def.get_components.return_value = [comp]

    ModelSaver.save_model(
        model=_make_wrapper(),
        bits={"transformer": 8},  # no "vae" key
        base_path=str(tmp_path),
        weight_definition=weight_def,
    )
    shard = next((tmp_path / "vae").glob("*.safetensors"))
    md = _read_shard_metadata(shard)
    assert str(md.get("quantization_level")) == "bf16"
