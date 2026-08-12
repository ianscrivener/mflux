# Z-Image NVFP4 Save Plan

## Goal

Support this Z-Image-only save workflow:

```bash
mflux-save \
  --model Tongyi-MAI/Z-Image-Turbo \
  --quantize nvfp4 \
  --path /Volumes/Extreme2Tb/MFlux-Models/Tongyi-MAI--Z-Image-Turbo-MFlux-nvfp4
```

The transformer and text encoder use MLX NVFP4. The VAE remains bf16.

## Constraints

- Keep the existing integer modes (`3`, `4`, `5`, `6`, `8`) behavior unchanged.
- Scope `nvfp4` acceptance to `mflux-save`; do not silently enable it for every generation command.
- Reject `nvfp4` for non-Z-Image models with a clear error until each model is explicitly validated.
- Preserve compatibility with existing mflux checkpoints whose `quantization_level` metadata is an integer.
- Use MLX `nn.quantize(mode="nvfp4", group_size=16, bits=4, quantize_input=True)` only for `nn.Linear` layers. NVFP4 activation quantization cannot be used for embeddings.

## Implementation

1. Introduce a shared quantization-level type that supports the existing integer modes plus the `"nvfp4"` sentinel. Update resolution, loaded metadata, image metadata, and display formatting to avoid integer-only parsing or `-bit` wording for NVFP4.
2. Let the save parser accept `nvfp4` without widening the normal generation parser. In `mflux-save`, validate that it is used only with Z-Image or Z-Image Turbo.
3. Extend `WeightApplier` with an NVFP4 branch. It must quantize eligible linear layers with MLX's NVFP4 parameters and leave all other modules intact; existing affine integer quantization remains on its current path.
4. Add a per-component quantization exclusion policy to `ComponentDefinition`. Configure only Z-Image's VAE to exclude `nvfp4`; do not set the existing global `skip_quantization` flag, because that would change integer-mode behavior.
5. Save metadata per component. The VAE shards/index carry no quantization level (bf16), while transformer and text-encoder shards/index carry `"nvfp4"`. On load, infer the stored mode from the quantized components and apply the component exclusion policy before updating VAE weights.
6. Update the Z-Image documentation and generated completion documentation so the option and VAE exception are discoverable.

## Tests

1. Parser tests: `mflux-save --quantize nvfp4` parses; a normal generation parser still rejects it; non-Z-Image save fails with the intended validation error.
2. Unit tests: NVFP4 selects `nn.Linear` with `mode="nvfp4"`, group size `16`, bits `4`, and quantized inputs; affine modes retain their current calls.
3. Save/load metadata tests: VAE records bf16/unquantized metadata while Z-Image transformer/text-encoder record `nvfp4`; integer checkpoint metadata still loads.
4. Slow integration test: save a Z-Image Turbo NVFP4 checkpoint, reload it without `--quantize`, verify VAE is not a quantized module, and generate a small deterministic image.
5. Run focused tests with `MFLUX_PRESERVE_TEST_OUTPUT=1`, then `make lint` and `make test-fast`. Verify MLX NVFP4 support on the minimum supported macOS and Linux MLX versions; raise the Linux minimum or return a clear capability error if needed.

## Files Expected To Change

- `src/mflux/cli/parser/parsers.py`
- `src/mflux/models/common/cli/save.py`
- `src/mflux/models/common/resolution/quantization_resolution.py`
- `src/mflux/models/common/weights/loading/{loaded_weights.py,weight_applier.py,weight_loader.py,weight_definition.py}`
- `src/mflux/models/common/weights/saving/model_saver.py`
- `src/mflux/models/z_image/weights/z_image_weight_definition.py`
- Z-Image/CLI completion docs and focused parser/model-saving tests

## GitNexus Assessment

- `CommandLineParser`: CRITICAL blast radius, 32 direct importers. The plan confines syntax expansion to the save path.
- `ModelSaver` and `WeightApplier`: shared cross-model infrastructure. Changes must be mode-gated, component-aware, and protected by backward-compatibility tests.