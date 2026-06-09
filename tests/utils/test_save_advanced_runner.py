"""Tests for the per-component bits-dict builder in save_advanced_runner."""
import argparse

import pytest

from mflux.utils.save_advanced_runner import build_quantize_dict, normalize_bits, validate_args


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
