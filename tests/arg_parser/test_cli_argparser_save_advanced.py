"""Tests for the mflux-save-advanced CLI argument parser."""
import sys
from unittest.mock import patch

import pytest

from mflux.cli.parser.parsers import CommandLineParser


@pytest.fixture
def parser() -> CommandLineParser:
    p = CommandLineParser(description="Save a model with per-component quantization")
    p.add_general_arguments()
    p.add_save_advanced_arguments()
    return p


def test_requires_one_model_flag(parser):
    with patch.object(sys, "argv", ["mflux-save-advanced", "--path", "/out"]):
        with pytest.raises(SystemExit):
            parser.parse_args()


def test_minimal_bf16_with_transformer_quant(parser, tmp_path):
    with patch.object(sys, "argv", [
        "mflux-save-advanced",
        "--bf16_model", str(tmp_path),
        "--path", "/out",
        "--quantize_transformer", "8",
    ]):
        args = parser.parse_args()
    assert args.bf16_model == str(tmp_path)
    assert args.quantize_transformer == "8"
    assert args.quantize_vae is None
    assert args.quantize_text_encoder is None


def test_mutually_exclusive(parser, tmp_path):
    with patch.object(sys, "argv", [
        "mflux-save-advanced",
        "--bf16_model", str(tmp_path),
        "--nvfp4_model", str(tmp_path),
        "--path", "/out",
    ]):
        with pytest.raises(SystemExit):
            parser.parse_args()


def test_force_default_false(parser, tmp_path):
    with patch.object(sys, "argv", [
        "mflux-save-advanced",
        "--bf16_model", str(tmp_path),
        "--path", "/out",
    ]):
        args = parser.parse_args()
    assert args.force is False


def test_force_flag(parser, tmp_path):
    with patch.object(sys, "argv", [
        "mflux-save-advanced",
        "--bf16_model", str(tmp_path),
        "--path", "/out",
        "--force",
    ]):
        args = parser.parse_args()
    assert args.force is True


def test_bf16_sentinel_accepted(parser, tmp_path):
    with patch.object(sys, "argv", [
        "mflux-save-advanced",
        "--bf16_model", str(tmp_path),
        "--path", "/out",
        "--quantize_vae", "bf16",
        "--quantize_transformer", "4",
    ]):
        args = parser.parse_args()
    assert args.quantize_vae == "bf16"
    assert args.quantize_transformer == "4"


def test_invalid_quantize_choice_rejected(parser, tmp_path):
    with patch.object(sys, "argv", [
        "mflux-save-advanced",
        "--bf16_model", str(tmp_path),
        "--path", "/out",
        "--quantize_vae", "7",  # 7 is not in the choices list
    ]):
        with pytest.raises(SystemExit):
            parser.parse_args()


def test_path_is_required(parser, tmp_path):
    with patch.object(sys, "argv", [
        "mflux-save-advanced",
        "--bf16_model", str(tmp_path),
    ]):
        with pytest.raises(SystemExit):
            parser.parse_args()


def test_nvfp4_path(parser, tmp_path):
    with patch.object(sys, "argv", [
        "mflux-save-advanced",
        "--nvfp4_model", str(tmp_path),
        "--path", "/out",
        "--force",
    ]):
        args = parser.parse_args()
    assert args.nvfp4_model == str(tmp_path)
    assert args.bf16_model is None
