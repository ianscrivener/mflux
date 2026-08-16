import pytest

from mflux.models.boogu.model.boogu_transformer.boogu_transformer import BooguImageTransformer
from mflux.models.boogu.weights.boogu_weight_definition import BooguWeightDefinition
from mflux.models.flux2.model.flux2_text_encoder.qwen3_text_encoder import Qwen3TextEncoder
from tests.model_saving.tiny_checkpoint_helper import TinyCheckpointRoundtrip, TinyVAEStandIn


class TestTinyBooguModelSaving:
    @pytest.mark.fast
    def test_tiny_quantized_checkpoint_roundtrips_exactly(self, tmp_path):
        # No slow twin exists for Boogu yet (no test_model_saving_boogu.py under
        # tests/model_saving/) — this is the first checkpoint coverage for it. Same
        # ModelSaver -> WeightLoader -> WeightApplier seam as the other tiny tests,
        # with real BooguImageTransformer/Qwen3TextEncoder component classes at tiny
        # dimensions instead of the downloaded checkpoint.
        TinyCheckpointRoundtrip.save_and_reload_expecting_identical_weights(
            weight_definition=BooguWeightDefinition,
            make_components=TestTinyBooguModelSaving._tiny_components,
            base_path=tmp_path / "boogu_tiny_q8",
            bits=8,
            # Force shard boundaries so index.json/weight_map multi-shard paths are
            # exercised — the size-based split never shards test-sized tensors.
            tensors_per_shard=8,
        )

    @staticmethod
    def _tiny_components():
        # BooguWeightDefinition.quantization_group_size is 32, not MLX's default 64
        # (production hidden_size 3360 is divisible by 32 but not 64). Most dimensions
        # below are also clean multiples of 64, but instruction_feat_dim=96 is not —
        # it exercises the non-default group size for real (fails to quantize at 64).
        # head_dim = hidden_size / num_attention_heads = 128 / 2 = 64.
        # axes_dim_rope must sum to head_dim: 16 + 24 + 24 = 64.
        # num_kv_heads (1) must divide num_attention_heads (2).
        # num_layers=4 (2 double + 2 single) with num_refiner_layers=2 gives every
        # repeated layer list at least 2 entries, exercising per-index key naming.
        return {
            "vae": TinyVAEStandIn(),
            "transformer": BooguImageTransformer(
                patch_size=2,
                in_channels=16,
                hidden_size=128,
                num_layers=4,
                num_double_stream_layers=2,
                num_refiner_layers=2,
                num_attention_heads=2,
                num_kv_heads=1,
                axes_dim_rope=(16, 24, 24),
                instruction_feat_dim=96,
            ),
            "text_encoder": Qwen3TextEncoder(
                vocab_size=128,
                hidden_size=128,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=64,
                intermediate_size=128,
            ),
        }
