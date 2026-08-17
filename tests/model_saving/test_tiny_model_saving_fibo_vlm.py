import pytest

from mflux.models.common_models.qwen3_vl.qwen3_vl_decoder import Qwen3VLDecoder
from mflux.models.common_models.qwen3_vl.qwen3_vl_vision_model import Qwen3VLVisionModel
from mflux.models.fibo_vlm.weights.fibo_vlm_weight_definition import FIBOVLMWeightDefinition
from tests.model_saving.tiny_checkpoint_helper import TinyCheckpointRoundtrip


class TestTinyFiboVlmModelSaving:
    @pytest.mark.fast
    def test_tiny_quantized_checkpoint_roundtrips_exactly(self, tmp_path):
        # No slow twin exists yet for fibo_vlm (no tests/model_saving/test_model_saving_fibo_vlm.py) —
        # this is the ModelSaver -> WeightLoader -> WeightApplier seam exercised directly with real
        # FIBO VLM component classes at tiny dimensions instead of a downloaded full-size checkpoint.
        TinyCheckpointRoundtrip.save_and_reload_expecting_identical_weights(
            weight_definition=FIBOVLMWeightDefinition,
            make_components=TestTinyFiboVlmModelSaving._tiny_components,
            base_path=tmp_path / "fibo_vlm_tiny_q8",
            bits=8,
            # Force shard boundaries so index.json/weight_map multi-shard paths are
            # exercised — the size-based split never shards test-sized tensors.
            tensors_per_shard=8,
        )

    @staticmethod
    def _tiny_components():
        # Every quantized Linear/Embedding last-dim is a multiple of 64 (MLX's default
        # quantization group size; fibo_vlm has no quantization_group_size override).
        # head_dim=64 satisfies both q/k/v/o_proj last-dims (128) and the RMSNorm sizes.
        # num_key_value_heads=1 divides num_attention_heads=2. mrope_section=[12, 10, 10]
        # is scaled down from the default [24, 20, 20] to stay within head_dim/2 (=32).
        # Qwen3VLVisionPatchMerger derives hidden_size = hidden_size * spatial_merge_size**2
        # = 128 * 4 = 512, itself a multiple of 64, so the merger's Linear layers quantize too.
        visual = Qwen3VLVisionModel(
            hidden_size=128,
            num_heads=2,
            intermediate_size=128,
            depth=2,
            spatial_merge_size=2,
            out_hidden_size=128,
        )

        # GOTCHA: `visual` is not an independent component. Qwen3VLDecoder.__init__ stores
        # whatever is passed as `visual=` as self.visual, and production's
        # FiboVLMInitializer._apply_weights registers the *same object* under both
        # component names: {"decoder": model.decoder, "visual": model.decoder.visual}.
        # We replicate that here by building the vision model once, threading it into the
        # decoder, and reading the "visual" entry back out via decoder.visual rather than
        # constructing a second, independent vision model — anything else would diverge
        # from the real save/load path this test is meant to exercise.
        #
        # This aliasing also matters at *save* time: FiboVLM only ever sets `.decoder` on
        # the real model object (see FiboVLMInitializer._init_models), never a separate
        # `.visual` attribute, so ModelSaver.save_model only ever writes decoder's tree
        # (which contains visual nested inside it) once. TinyCheckpointRoundtrip._as_model
        # mirrors that: it skips assigning a model attribute for any component whose module
        # is already reachable inside another component's tree, so this dict can still hand
        # WeightApplier both aliases (matching _apply_weights) without ModelSaver writing
        # "visual" a second time into the same hf_subdir and clobbering decoder's shards.
        decoder = Qwen3VLDecoder(
            vocab_size=128,
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            intermediate_size=128,
            head_dim=64,
            mrope_section=[12, 10, 10],
            visual=visual,
        )

        return {
            "decoder": decoder,
            "visual": decoder.visual,
        }
