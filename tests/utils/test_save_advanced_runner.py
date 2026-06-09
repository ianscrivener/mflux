"""Tests for the per-component bits-dict builder in save_advanced_runner."""
import argparse

import pytest

from mflux.models.flux.variants.txt2img.flux import Flux1
from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein
from mflux.models.z_image import ZImage
from mflux.utils.save_advanced_runner import (
    _expand_quantize_dict,
    _infer_model_from_path,
    build_quantize_dict,
    normalize_bits,
    validate_args,
)


def _args(**kwargs):
    a = argparse.Namespace(
        bf16_model=None,
        nvfp4_model=None,
        path="/out",
        force=False,
    )
    a.__dict__.update(kwargs)
    return a


def test_normalize_bits_bf16_returns_none():
    assert normalize_bits("bf16") is None


def test_normalize_bits_int_string():
    assert normalize_bits("8") == 8
    assert normalize_bits("4") == 4


def test_normalize_bits_passthrough_int():
    assert normalize_bits(8) == 8


def test_normalize_bits_none_passthrough():
    assert normalize_bits(None) is None


def test_build_quantize_dict_with_string_bf16():
    a = _args(quantize_vae="bf16", quantize_transformer="8")
    assert build_quantize_dict(a) == {"vae": None, "transformer": 8}


def test_build_quantize_dict_skips_missing():
    a = _args(quantize_transformer="4")
    assert build_quantize_dict(a) == {"transformer": 4}


def test_build_quantize_dict_empty():
    a = _args()
    assert build_quantize_dict(a) == {}


def test_validate_args_rejects_both_flags(tmp_path):
    a = _args(bf16_model=str(tmp_path), nvfp4_model=str(tmp_path))
    with pytest.raises(SystemExit):
        validate_args(a)


def test_validate_args_rejects_neither_flag():
    a = _args()
    with pytest.raises(SystemExit):
        validate_args(a)


def test_expand_flux1_splits_text_encoder():
    """Flux1 has t5_encoder + clip_encoder; the CLI's single text_encoder
    flag must expand to both with the same bits value."""
    out = _expand_quantize_dict({"text_encoder": 8}, Flux1)
    assert out == {"t5_encoder": 8, "clip_encoder": 8}


def test_expand_flux1_with_vae_and_transformer():
    out = _expand_quantize_dict(
        {"vae": 4, "transformer": 8, "text_encoder": None},
        Flux1,
    )
    assert out == {"vae": 4, "transformer": 8, "t5_encoder": None, "clip_encoder": None}


def test_expand_flux2_text_encoder_passes_through():
    out = _expand_quantize_dict({"text_encoder": 8}, Flux2Klein)
    assert out == {"text_encoder": 8}


def test_expand_z_image_text_encoder_passes_through():
    out = _expand_quantize_dict({"text_encoder": 4}, ZImage)
    assert out == {"text_encoder": 4}


def test_infer_model_flux_path():
    assert _infer_model_from_path("black-forest-labs/FLUX.1-dev") == "dev"
    assert _infer_model_from_path("/local/path/to/dev") == "dev"


def test_infer_model_flux2_path():
    assert _infer_model_from_path("black-forest-labs/FLUX.2-klein-4B") == "flux2-klein-base-4b"


def test_infer_model_qwen_path():
    assert _infer_model_from_path("Qwen/Qwen-Image") == "qwen-image"
    assert _infer_model_from_path("Qwen/Qwen-Image-Edit") == "qwen-image-edit"


def test_infer_model_fibo_path():
    assert _infer_model_from_path("some/fibo-repo") == "fibo"
