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

from mflux.models.common.cli.save import resolve_model_class
from mflux.utils.save_overwrite import confirm_overwrite
from mflux.utils.source_format import (
    detect_bf16_pytorch_safetensors,
    detect_nvfp4_pytorch_safetensors,
)


# Component keys exposed at the CLI. Note: the architecture-specific key
# translation (e.g. "text_encoder" -> ["t5_encoder", "clip_encoder"] for Flux1
# vs "text_encoder" for Flux2) happens later, via _expand_quantize_dict.
COMPONENT_KEYS: tuple[str, ...] = ("vae", "transformer", "text_encoder")

# Per-architecture expansion of the CLI's `text_encoder` key into the actual
# model attribute names. Flux1 splits text encoding into t5_encoder +
# clip_encoder; every other supported architecture has a single text_encoder.
TEXT_ENCODER_EXPANSION: dict[type, tuple[str, ...]] = {}  # populated lazily


def _get_text_encoder_components(model_class: type) -> tuple[str, ...]:
    """Return the model-attribute name(s) for the CLI's `text_encoder` key."""
    # Lazy import to avoid pulling every model class at module load time.
    if not TEXT_ENCODER_EXPANSION:
        from mflux.models.flux.variants.txt2img.flux import Flux1
        from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein
        from mflux.models.z_image import ZImage
        from mflux.models.fibo.variants.txt2img.fibo import FIBO
        from mflux.models.ernie_image.variants.txt2img.ernie_image import ErnieImage
        from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage
        TEXT_ENCODER_EXPANSION.update({
            Flux1: ("t5_encoder", "clip_encoder"),
            Flux2Klein: ("text_encoder",),
            ZImage: ("text_encoder",),
            FIBO: ("text_encoder",),
            ErnieImage: ("text_encoder",),
            QwenImage: ("text_encoder",),
        })
    return TEXT_ENCODER_EXPANSION.get(model_class, ("text_encoder",))


def _expand_quantize_dict(
    cli_dict: dict[str, int | None],
    model_class: type,
) -> dict[str, int | None]:
    """Expand the CLI's `text_encoder` key into the architecture's actual
    model attribute names. For Flux1, the same bits value is applied to both
    t5_encoder and clip_encoder (the CLI's text encoder granularity is wider
    than the architecture's).
    """
    out: dict[str, int | None] = {}
    for key, value in cli_dict.items():
        if key == "text_encoder":
            for expanded in _get_text_encoder_components(model_class):
                out[expanded] = value
        else:
            out[key] = value
    return out


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


def _infer_model_from_path(path: str) -> str:
    """Best-effort model-name inference from a HuggingFace repo or local path.

    Used when the user did not supply --model. Looks at the trailing segment
    of a 'org/name' HF id and matches it against known patterns.
    """
    p = Path(path)
    name = p.name.lower()
    if "flux2" in name or "flux.2" in name or "klein" in name:
        return "flux2-klein-base-4b"
    if "ernie" in name:
        return "ernie-image"
    if "qwen" in name and "edit" in name:
        return "qwen-image-edit"
    if "qwen" in name:
        return "qwen-image"
    if "fibo" in name:
        return "fibo"
    if "z-image-turbo" in name or "zimage-turbo" in name:
        return "z-image-turbo"
    if "z-image" in name or "zimage" in name:
        return "z-image"
    if "ideogram" in name:
        return "ideogram4"
    # Default: FLUX.1 dev (most common bf16 pytorch source on HF)
    return "dev"


def _resolve_args_model(args: argparse.Namespace, source_path: str) -> argparse.Namespace:
    """Ensure args has a `model` attribute for resolve_model_class.

    If the user did not pass --model, infer from the source path. We mutate
    a shallow copy of args so the original Namespace is not changed (avoids
    surprising the caller).
    """
    if getattr(args, "model", None):
        return args
    ns = argparse.Namespace(**vars(args))
    ns.model = _infer_model_from_path(source_path)
    return ns


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


def _build_model_for_bf16(args: argparse.Namespace, model_class: type):
    """Construct the model instance for the bf16 path.

    Mirrors mflux-save's construction logic but with a per-component
    quantize dict instead of a single int.
    """
    from mflux.models.common.config import ModelConfig
    from mflux.models.ideogram4.weights.ideogram4_weight_definition import (
        Ideogram4WeightDefinition,
    )
    from mflux.models.ideogram4.variants.txt2img.ideogram4 import Ideogram4

    cli_quantize = build_quantize_dict(args)
    model_quantize = _expand_quantize_dict(cli_quantize, model_class)

    if model_class is Ideogram4:
        model_config = Ideogram4WeightDefinition.resolve_inference_config(
            args.model, args.base_model, args.bf16_model,
        )
        model_path = None if Ideogram4WeightDefinition.is_builtin_name(args.model) else args.bf16_model
    else:
        model_config = ModelConfig.from_name(args.model, base_model=args.base_model)
        model_path = args.bf16_model

    return model_class(
        quantize=model_quantize,
        model_path=model_path,
        model_config=model_config,
    )


def run_bf16(args: argparse.Namespace) -> None:
    """Convert a bf16 pytorch model with per-component quantization."""
    effective_args = _resolve_args_model(args, args.bf16_model)
    model_class = resolve_model_class(effective_args)
    model = _build_model_for_bf16(effective_args, model_class)
    model.save_model(args.path)


def run_nvfp4(args: argparse.Namespace) -> None:
    """Convert the transformer of an nvfp4 pytorch model.

    This path is a substantial piece of work (NVFP4 pytorch parser, dequant
    pipeline, NVFP4-aware saver) that depends on a separate branch of work
    in the user's fork. The CLI surfaces a clear message rather than silently
    doing the wrong thing.
    """
    raise NotImplementedError(
        "mflux-save-advanced: --nvfp4_model conversion is not yet implemented "
        "in this branch. The CLI is wired up to accept the flag and validate "
        "the source format, but the transformer-only NVFP4 conversion is "
        "tracked as a separate piece of work (BRANCH_README Task 2)."
    )


def main(args: argparse.Namespace) -> None:
    """End-to-end entry point for the CLI module."""
    validate_args(args)
    confirm_overwrite(Path(args.path), force=args.force)
    if args.bf16_model:
        run_bf16(args)
    else:
        run_nvfp4(args)
