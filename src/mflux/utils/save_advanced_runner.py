"""Top-level dispatch for mflux-save-advanced.

This module is the implementation behind the new mflux-save-advanced console
script. It validates the source format, prompts for overwrite, builds the
per-component quantization dict, and routes to the bf16 or nvfp4 save path.

The bf16 path (Task 1 in the PRD) supports per-component quantization with
optional flags: --quantize_vae, --quantize_transformer, --quantize_text_encoder.
The nvfp4 path (Task 2) currently saves the transformer only.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from mflux.utils.save_overwrite import confirm_overwrite
from mflux.utils.source_format import (
    detect_bf16_pytorch_safetensors,
    detect_nvfp4_pytorch_safetensors,
)


# Component keys exposed at the CLI. Note: the architecture-specific key
# translation (e.g. "text_encoder" -> ["t5_encoder", "clip_encoder"] for Flux1
# vs "text_encoder" for Flux2) happens later, in the model wrapper.
COMPONENT_KEYS: tuple[str, ...] = ("vae", "transformer", "text_encoder")


def normalize_bits(value: str | int | None) -> int | None:
    """Map a CLI value to an internal bits representation.

    - "bf16"  -> None  (no quantization)
    - "3".."8" or 3..8 -> int(value)
    - None    -> None
    """
    if value is None or value == "bf16":
        return None
    return int(value)


def build_quantize_dict(args: argparse.Namespace) -> dict[str, int | None]:
    """Build a {component: bits} dict from the CLI args.

    Components whose flag is not provided are simply absent from the dict,
    which is what makes the per-component save flow work (see
    WeightApplier._apply_from_cli_dict).
    """
    out: dict[str, int | None] = {}
    for key in COMPONENT_KEYS:
        raw = getattr(args, f"quantize_{key}", None)
        if raw is not None:
            out[key] = normalize_bits(raw)
    return out


def validate_args(args: argparse.Namespace) -> None:
    """Reject combinations that the parser can't catch (and surface clear
    errors for format mismatches)."""
    if args.bf16_model and args.nvfp4_model:
        raise SystemExit(
            "mflux-save-advanced: use either --bf16_model or --nvfp4_model, not both."
        )
    if not args.bf16_model and not args.nvfp4_model:
        # argparse's mutex group should already prevent this, but be defensive.
        raise SystemExit(
            "mflux-save-advanced: one of --bf16_model or --nvfp4_model is required."
        )
    if args.bf16_model:
        if not detect_bf16_pytorch_safetensors(args.bf16_model):
            raise SystemExit(
                f"mflux-save-advanced: source {args.bf16_model} is not a bf16 "
                "pytorch-safetensors model. mflux-save-advanced currently only "
                "converts bf16 pytorch format."
            )
    elif args.nvfp4_model:
        if not detect_nvfp4_pytorch_safetensors(args.nvfp4_model):
            raise SystemExit(
                f"mflux-save-advanced: source {args.nvfp4_model} is not a "
                "nvfp4 pytorch-safetensors model. mflux-save-advanced currently "
                "only converts nvfp4 pytorch format."
            )


def run_bf16(args: argparse.Namespace) -> None:
    """Convert a bf16 pytorch model with per-component quantization.

    Wired up in Task 11.
    """
    raise NotImplementedError


def run_nvfp4(args: argparse.Namespace) -> None:
    """Convert the transformer of an nvfp4 pytorch model.

    Wired up in Task 12.
    """
    raise NotImplementedError


def main(args: argparse.Namespace) -> None:
    """End-to-end entry point for the CLI module."""
    validate_args(args)
    confirm_overwrite(Path(args.path), force=args.force)
    if args.bf16_model:
        run_bf16(args)
    else:
        run_nvfp4(args)
