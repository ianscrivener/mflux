import mlx.core as mx
import pytest
from mlx import nn

from mflux.callbacks.instances.memory_saver import MemorySaver
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.tokenizer.tokenizer_output import TokenizerOutput
from mflux.models.z_image.variants.z_image import ZImage


class _Tokenizer:
    max_length = 8
    use_chat_template = True
    chat_template_kwargs = {"enable_thinking": True}
    template = None

    def tokenize(self, prompt, **kwargs):
        return TokenizerOutput(input_ids=mx.zeros((1, 8), dtype=mx.int32), attention_mask=mx.ones((1, 8)))


class _Encoder:
    def __init__(self):
        self.calls = 0

    def __call__(self, input_ids, attention_mask):
        self.calls += 1
        return mx.full((1, input_ids.shape[1], 4), float(self.calls))


def _model(small: bool = False, bits: int | None = None) -> tuple[ZImage, _Encoder]:
    model = ZImage.__new__(ZImage)
    nn.Module.__init__(model)
    encoder = _Encoder()
    model.model_path = None
    model.model_config = ModelConfig.z_image()
    model.tokenizers = {"z_image": _Tokenizer()}
    model.text_encoder = encoder
    model.small_text_encoder = small
    model.bits = bits
    return model, encoder


@pytest.mark.fast
def test_repeated_prompt_is_encoded_once():
    model, encoder = _model()
    first, _ = model._encode_prompts(prompt="a cat", negative_prompt=None, guidance=0.0)
    second, _ = model._encode_prompts(prompt="a cat", negative_prompt=None, guidance=0.0)
    assert encoder.calls == 1
    assert mx.array_equal(first, second).item()


@pytest.mark.fast
def test_empty_negative_prompt_is_cached_like_any_prompt():
    model, encoder = _model()
    model._encode_prompts(prompt="a cat", negative_prompt=None, guidance=4.0)
    model._encode_prompts(prompt="a cat", negative_prompt="", guidance=4.0)
    assert encoder.calls == 2  # "a cat" and " " once each


@pytest.mark.fast
def test_cache_evicts_the_least_recently_used_prompt():
    model, encoder = _model()
    for i in range(ZImage.PROMPT_CACHE_SIZE):
        model._encode_prompts(prompt=f"p{i}", negative_prompt=None, guidance=0.0)
    model._encode_prompts(prompt="p0", negative_prompt=None, guidance=0.0)  # p0 is now the newest
    model._encode_prompts(prompt="new", negative_prompt=None, guidance=0.0)  # evicts p1
    assert model.has_cached_prompt("p0")
    assert not model.has_cached_prompt("p1")


@pytest.mark.fast
def test_key_separates_encoder_variant_and_bits():
    a, _ = _model(small=False, bits=None)
    b, _ = _model(small=True, bits=None)
    c, _ = _model(small=False, bits=8)
    assert len({a._prompt_cache_key("x"), b._prompt_cache_key("x"), c._prompt_cache_key("x")}) == 3


@pytest.mark.fast
def test_cache_is_not_a_model_parameter():
    model, _ = _model()
    model._encode_prompts(prompt="a cat", negative_prompt=None, guidance=0.0)
    assert "_prompt_cache" not in model.parameters()


@pytest.mark.fast
def test_memory_saver_drops_encoder_for_multi_seed_when_prompt_is_cached():
    model, _ = _model()
    model._encode_prompts(prompt="a cat", negative_prompt=None, guidance=0.0)
    saver = MemorySaver(model, cache_limit_bytes=None, num_seeds=3)
    saver.call_before_loop(seed=1, prompt="a cat", latents=mx.zeros((1,)), config=None)
    assert model.text_encoder is None


@pytest.mark.fast
def test_timing_is_readable_and_reset_when_disabled(monkeypatch):
    model, _ = _model()
    monkeypatch.setenv("MFLUX_TIMING", "1")
    model._encode_prompts(prompt="a cat", negative_prompt=None, guidance=0.0)
    assert model.timing.num_valid == 8 and model.timing.prefill_s >= 0
    assert "timing" not in model.parameters()
    monkeypatch.delenv("MFLUX_TIMING")
    model._encode_prompts(prompt="a cat", negative_prompt=None, guidance=0.0)
    assert model.timing is None


@pytest.mark.fast
def test_uncached_prompt_without_text_encoder_raises_a_clear_error():
    model, _ = _model()
    model._encode_prompts(prompt="a cat", negative_prompt=None, guidance=0.0)
    model.text_encoder = None
    model._encode_prompts(prompt="a cat", negative_prompt=None, guidance=0.0)  # a hit still works
    with pytest.raises(RuntimeError, match="text encoder is released"):
        model._encode_prompts(prompt="a dog", negative_prompt=None, guidance=0.0)
