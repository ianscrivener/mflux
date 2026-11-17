import mlx.core as mx

from mflux.models.common.config import ModelConfig
from mflux.models.z_image.model.z_image_text_encoder.text_encoder import TextEncoder


def _tiny_encoder(layers: int = 4) -> TextEncoder:
    mx.random.seed(0)
    encoder = TextEncoder(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=64,
        head_dim=8,
    )
    mx.eval(encoder.parameters())
    return encoder


def _reference_output(encoder: TextEncoder, input_ids: mx.array, attention_mask: mx.array) -> mx.array:
    # The old behaviour: run every layer and return hidden_states[-2].
    batch_size, seq_len = input_ids.shape
    hidden_states = encoder.embed_tokens(input_ids).astype(mx.float32)
    position_ids = mx.broadcast_to(mx.arange(seq_len, dtype=mx.int32)[None, :], (batch_size, seq_len))
    position_embeddings = encoder.rotary_emb(hidden_states, position_ids)
    mask = TextEncoder._get_causal_mask(attention_mask, batch_size, hidden_states, seq_len)
    all_hidden_states = [hidden_states]
    for layer in encoder.layers:
        hidden_states = layer(hidden_states=hidden_states, attention_mask=mask, position_embeddings=position_embeddings)
        all_hidden_states.append(hidden_states)
    return all_hidden_states[-2].astype(ModelConfig.precision)


def test_early_stop_is_bit_identical_to_full_loop():
    encoder = _tiny_encoder()
    input_ids = mx.array([[1, 2, 3, 4, 5, 0, 0, 0]])
    attention_mask = mx.array([[1, 1, 1, 1, 1, 0, 0, 0]])
    actual = encoder(input_ids, attention_mask)
    expected = _reference_output(encoder, input_ids, attention_mask)
    assert mx.array_equal(actual, expected).item()


def test_last_layer_weights_do_not_affect_the_output():
    encoder = _tiny_encoder()
    input_ids = mx.array([[1, 2, 3]])
    attention_mask = mx.array([[1, 1, 1]])
    before = encoder(input_ids, attention_mask)
    encoder.layers[-1].self_attn.q_proj.weight = mx.full(encoder.layers[-1].self_attn.q_proj.weight.shape, mx.nan)
    assert mx.array_equal(encoder(input_ids, attention_mask), before).item()


def _compare_padded_and_trimmed(model, vocab: int, num_valid: int = 5, padded: int = 16) -> None:
    ids = mx.random.randint(1, vocab, (1, num_valid))
    padded_ids = mx.concatenate([ids, mx.zeros((1, padded - num_valid), dtype=ids.dtype)], axis=1)
    padded_mask = mx.array([[1] * num_valid + [0] * (padded - num_valid)])
    full = model(padded_ids, padded_mask)[0, :num_valid].astype(mx.float32)
    trimmed = model(padded_ids[:, :num_valid], padded_mask[:, :num_valid])[0].astype(mx.float32)
    assert mx.max(mx.abs(full - trimmed)).item() < 1e-4
    cosine = mx.sum(full * trimmed) / (mx.linalg.norm(full) * mx.linalg.norm(trimmed))
    assert cosine.item() > 0.9999


def test_trimmed_input_matches_padded_input(monkeypatch):
    monkeypatch.setattr(ModelConfig, "precision", mx.float32)
    _compare_padded_and_trimmed(_tiny_encoder(), vocab=64)


def test_trimmed_input_matches_padded_input_with_adapter(monkeypatch):
    from mflux.models.z_image.model.z_image_text_encoder.small_text_encoder import ZImageSmallTextEncoder
    from mflux.models.z_image.model.z_image_text_encoder.student_adapter import ZImageStudentAdapter

    monkeypatch.setattr(ModelConfig, "precision", mx.float32)
    mx.random.seed(1)
    adapter = ZImageStudentAdapter(student_dim=32, teacher_dim=48, dim=16, heads=2, blocks=2, ff_mult=2)
    mx.eval(adapter.parameters())
    _compare_padded_and_trimmed(ZImageSmallTextEncoder(student=_tiny_encoder(), adapter=adapter), vocab=64)
