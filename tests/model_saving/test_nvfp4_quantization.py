import mlx.core as mx
import mlx.nn as nn

from mflux.models.common.weights.loading.weight_applier import WeightApplier
from mflux.models.common.weights.loading.weight_loader import WeightLoader
from mflux.models.common.weights.saving.model_saver import ModelSaver
from mflux.models.z_image.weights.z_image_weight_definition import ZImageWeightDefinition


class TinyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(16, 16, bias=False)
        self.biased_linear = nn.Linear(16, 16)
        self.embedding = nn.Embedding(4, 16)


class TinyZImage:
    def __init__(self):
        self.tokenizers = {}
        self.vae = TinyModule()
        self.transformer = TinyModule()
        self.text_encoder = TinyModule()


def test_nvfp4_quantizes_linear_layers_and_preserves_vae():
    transformer = TinyModule()
    vae = TinyModule()
    components = {component.name: component for component in ZImageWeightDefinition.get_components()}

    WeightApplier._quantize(
        models={"transformer": transformer, "vae": vae},
        bits="nvfp4",
        components=components,
        weight_definition=ZImageWeightDefinition,
    )

    assert isinstance(transformer.linear, nn.QuantizedLinear)
    assert isinstance(transformer.biased_linear, nn.QuantizedLinear)
    assert isinstance(transformer.embedding, nn.Embedding)
    assert isinstance(vae.linear, nn.Linear)


def test_nvfp4_checkpoint_keeps_vae_metadata_unquantized(tmp_path):
    model = TinyZImage()
    components = {component.name: component for component in ZImageWeightDefinition.get_components()}
    WeightApplier._quantize(
        models={"transformer": model.transformer, "text_encoder": model.text_encoder, "vae": model.vae},
        bits="nvfp4",
        components=components,
        weight_definition=ZImageWeightDefinition,
    )
    ModelSaver.save_model(
        model=model,
        bits="nvfp4",
        base_path=str(tmp_path),
        weight_definition=ZImageWeightDefinition,
    )

    _, vae_quantization, _ = WeightLoader._try_load_mflux_format(tmp_path / "vae")
    transformer_weights, transformer_quantization, _ = WeightLoader._try_load_mflux_format(tmp_path / "transformer")
    loaded = WeightLoader.load(weight_definition=ZImageWeightDefinition, model_path=str(tmp_path))

    assert vae_quantization is None
    assert transformer_quantization == "nvfp4"
    assert transformer_weights["linear"]["weight"].dtype == mx.uint32
    assert loaded.meta_data.quantization_level == "nvfp4"


def test_model_saver_removes_stale_component_shards(tmp_path):
    model = TinyZImage()
    ModelSaver.save_model(
        model=model,
        bits="nvfp4",
        base_path=str(tmp_path),
        weight_definition=ZImageWeightDefinition,
    )
    stale_shard = tmp_path / "transformer" / "stale.safetensors"
    mx.save_safetensors(str(stale_shard), {"stale": mx.array([1])})

    ModelSaver.save_model(
        model=model,
        bits="nvfp4",
        base_path=str(tmp_path),
        weight_definition=ZImageWeightDefinition,
    )

    assert not stale_shard.exists()
