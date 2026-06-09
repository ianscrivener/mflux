from typing import TYPE_CHECKING

import mlx.core as mx
import mlx.nn as nn

from mflux.models.common.resolution.quantization_resolution import QuantizationResolution
from mflux.models.common.weights.loading.loaded_weights import LoadedWeights
from mflux.models.common.weights.loading.weight_definition import ComponentDefinition

if TYPE_CHECKING:
    from mflux.models.common.weights.loading.weight_definition import WeightDefinitionType


class WeightApplier:
    @staticmethod
    def apply_and_quantize_single(
        weights: LoadedWeights,
        model: nn.Module,
        component: ComponentDefinition,
        quantize_arg: int | str | None,
        quantization_predicate=None,
    ) -> int | str | None:
        stored_q = weights.meta_data.quantization_level
        component_weights = weights.components.get(component.name)

        if component_weights is None:
            raise ValueError(f"No weights found for component: {component.name}")

        if quantization_predicate is None:

            def quantization_predicate(path, module):
                return hasattr(module, "to_quantized")

        bits, warning = QuantizationResolution.resolve(stored=stored_q, requested=quantize_arg)

        if warning:
            print(f"⚠️  {warning}")

        if bits is None:
            model.update(component_weights, strict=False)
        elif stored_q is None:
            model.update(component_weights, strict=False)
            if not component.skip_quantization:
                WeightApplier._apply_quantize(model, bits, quantization_predicate)
        else:
            if not component.skip_quantization:
                WeightApplier._apply_quantize(model, bits, quantization_predicate)
            model.update(component_weights, strict=False)

        return bits

    @staticmethod
    def apply_and_quantize(
        weights: LoadedWeights,
        models: dict[str, nn.Module],
        quantize_arg: int | str | None,
        weight_definition: "WeightDefinitionType",
    ) -> int | str | None:
        components = {c.name: c for c in weight_definition.get_components()}

        # Per-component quantization level (falls back to the global
        # quantization_level for backward compatibility with callers that
        # constructed LoadedWeights manually).
        per_comp = weights.meta_data.per_component_quantization_level or {}
        component_q: dict[str, int | str | None] = {}
        any_quantize = False
        for name in models.keys():
            q = per_comp.get(name, weights.meta_data.quantization_level)
            component_q[name] = q
            if q is not None:
                any_quantize = True

        if not any_quantize and quantize_arg is None:
            # Nothing to quantize at all — just load.
            WeightApplier._set_weights(weights, models, components)
            return None

        # Resolve per-component. Each component independently decides whether
        # to load-then-quantize or quantize-then-load based on its own stored_q.
        for name, model in models.items():
            stored_q = component_q.get(name)
            component = components.get(name)
            component_weights = weights.components.get(name)

            bits, warning = QuantizationResolution.resolve(
                stored=stored_q, requested=quantize_arg,
            )
            if warning:
                print(f"⚠️  [{name}] {warning}")

            if bits is None:
                # No quantization for this component — just load.
                if component_weights is not None:
                    model.update(component_weights, strict=False)
            elif stored_q is None:
                # On-the-fly: load real weights, then quantize.
                if component_weights is not None:
                    model.update(component_weights, strict=False)
                if component is None or not component.skip_quantization:
                    WeightApplier._apply_quantize(
                        model, bits, weight_definition.quantization_predicate,
                    )
            else:
                # Pre-quantized: quantize the freshly-initialized model first,
                # then load the saved (already-quantized) weights.
                if component is None or not component.skip_quantization:
                    WeightApplier._apply_quantize(
                        model, bits, weight_definition.quantization_predicate,
                    )
                if component_weights is not None:
                    # Dequantize legacy embeddings if this component had a
                    # saved-format packed embedding but the in-memory model
                    # is no longer quantized (per the predicate).
                    component_weights = WeightApplier._dequantize_legacy_embeddings(
                        component_weights, stored_q,
                    )
                    model.update(component_weights, strict=False)

        return None

    @staticmethod
    def _set_weights(
        weights: LoadedWeights,
        models: dict[str, nn.Module],
        components: dict | None = None,
    ) -> None:
        for name, model in models.items():
            component_weights = weights.components.get(name)
            if component_weights is not None:
                if components is not None:
                    component = components.get(name)
                    if component is not None and component.weight_subkey is not None:
                        component_weights = component_weights.get(component.weight_subkey, component_weights)
                # Compatibility shim: a model saved with the pre-fix Flux2
                # predicate (which quantized nn.Embedding) has the embedding
                # saved as a packed uint32 weight + float32 scales (+ optional
                # biases). The current Flux2 predicate leaves the embedding as
                # a regular nn.Embedding with a float32 weight, so a naive
                # update() would silently overwrite the in-memory float32
                # weight with a uint32 packed array of the wrong shape,
                # producing a downstream shape mismatch in the first RMSNorm
                # (the error the user sees is "RMSNorm got (..., 320) but
                # weight is (2560,)").
                #
                # Detect this case and dequantize the saved embedding to a
                # plain float32 weight before update, so the in-memory
                # Embedding (which is no longer quantized) gets correct values.
                component_weights = WeightApplier._dequantize_legacy_embeddings(
                    component_weights, weights.meta_data.quantization_level,
                )
                model.update(component_weights, strict=False)

    @staticmethod
    def _dequantize_legacy_embeddings(component_weights: dict, quantization_level) -> dict:
        """If a saved embedding was packed uint32 (old save format) but the
        in-memory model has a regular nn.Embedding, dequantize the embedding
        in-place so the in-memory weight gets the correct float values.
        Returns a (possibly shallow-copied) dict."""
        if not isinstance(component_weights, dict):
            return component_weights
        # The saved format puts the Embedding at the top of the component's
        # state dict (e.g. component_weights["embed_tokens"]). Detect the
        # packed uint32 weight and dequantize it to float32.
        embed = component_weights.get("embed_tokens")
        if not isinstance(embed, dict):
            return component_weights
        saved_weight = embed.get("weight")
        if not isinstance(saved_weight, mx.array) or saved_weight.dtype != mx.uint32:
            return component_weights
        # Determine mode and group_size from the saved quantization level
        if quantization_level == "nvfp4":
            mode = "nvfp4"
            group_size = 16
            bits = 4
        elif isinstance(quantization_level, int):
            mode = "affine"
            group_size = 64
            bits = quantization_level
        else:
            return component_weights  # unknown — don't touch
        scales = embed.get("scales")
        biases = embed.get("biases")
        if not isinstance(scales, mx.array):
            return component_weights
        # Dequantize: (vocab, packed_dim) → (vocab, dim)
        dequantized = mx.dequantize(
            saved_weight,
            scales=scales,
            biases=biases,
            group_size=group_size,
            bits=bits,
            mode=mode,
        ).astype(mx.float32)
        # Build a new dict with the dequantized weight, dropping scales/biases
        # (which the in-memory Embedding doesn't have).
        new_embed = {k: v for k, v in embed.items() if k not in ("weight", "scales", "biases")}
        new_embed["weight"] = dequantized
        new_component_weights = dict(component_weights)
        new_component_weights["embed_tokens"] = new_embed
        return new_component_weights

    @staticmethod
    def _quantize(
        models: dict[str, nn.Module],
        bits: int | str,
        components: dict,
        weight_definition: "WeightDefinitionType",
    ) -> None:
        for name, model in models.items():
            component = components.get(name)
            if component and component.skip_quantization:
                continue
            WeightApplier._apply_quantize(model, bits, weight_definition.quantization_predicate)

    @staticmethod
    def _apply_quantize(model: nn.Module, bits: int | str, predicate) -> None:
        if bits == "nvfp4":
            # NVFP4 weight-only path. MLX's documented recommended use of nvfp4
            # is weight+activation quantization (QQLinear / mx.qqmm), but the
            # QQMatmul kernel is "NYI for the general case" on Mac/MLX 0.31.x
            # in this build, so we cannot use it. When the kernel becomes
            # available, switch the predicate to return
            # ``{"group_size": 16, "bits": 4, "mode": "nvfp4", "quantize_input": True}``
            # for bias-free Linear layers (QQLinear does not support bias —
            # see mlx/nn/layers/quantized.py:324, 420-421). The Flux2 weight
            # definition already filters out Embedding layers in its predicate.
            nn.quantize(model, class_predicate=predicate, mode="nvfp4")
        else:
            nn.quantize(model, class_predicate=predicate, bits=bits)
