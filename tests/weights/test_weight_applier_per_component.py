"""Tests for the per-component (dict) quantize_arg path in WeightApplier."""
from unittest.mock import MagicMock

from mlx import nn

from mflux.models.common.weights.loading.loaded_weights import LoadedWeights, MetaData
from mflux.models.common.weights.loading.weight_applier import WeightApplier


class TinyModel(nn.Module):
    """Minimal nn.Module with a Linear that tree_unflatten can walk."""

    def __init__(self):
        super().__init__()
        # 128-dim so quantization group_size=64 fits. The 2-d version below is
        # just to keep imports simple; both work for the non-quantizing path.
        self.linear = nn.Linear(128, 128)


def _build_weight_definition(*component_names: str):
    """Build a MagicMock weight definition that returns ComponentDefinition-like
    objects whose .skip_quantization is False and .weight_subkey is None.
    """
    wd = MagicMock()

    def _quant_predicate(path, module):
        return hasattr(module, "to_quantized")

    wd.quantization_predicate = _quant_predicate

    components = []
    for name in component_names:
        comp = MagicMock()
        comp.name = name
        comp.skip_quantization = False
        comp.weight_subkey = None
        components.append(comp)
    wd.get_components.return_value = components
    return wd


def _make_loaded_weights(*component_names: str, stored_q: int | None = None) -> LoadedWeights:
    """Build a LoadedWeights with one fake weight per component name."""
    import mlx.core as mx

    components = {
        name: {"w": mx.zeros((2, 2))}
        for name in component_names
    }
    return LoadedWeights(
        components=components,
        meta_data=MetaData(quantization_level=stored_q, component_quantization_levels={}),
    )


def test_int_quantize_arg_returns_int():
    """Legacy single-int path still works (regression guard)."""
    weights = _make_loaded_weights("vae", stored_q=None)
    wd = _build_weight_definition("vae")
    result = WeightApplier.apply_and_quantize(
        weights=weights,
        models={"vae": TinyModel()},
        quantize_arg=None,  # bf16
        weight_definition=wd,
    )
    assert result is None


def test_dict_quantize_arg_quantizes_specified_component():
    """Dict quantize_arg with a known int value quantizes that component and
    returns a per-component dict result."""
    weights = _make_loaded_weights("vae", stored_q=None)
    wd = _build_weight_definition("vae")
    result = WeightApplier.apply_and_quantize(
        weights=weights,
        models={"vae": TinyModel()},
        quantize_arg={"vae": 4, "transformer": 8},  # transformer is absent in models
        weight_definition=wd,
    )
    # Returns a per-component dict; the model bits becomes the per-component map.
    assert result == {"vae": 4}


def test_dict_quantize_arg_skips_missing_component():
    """Components missing from the dict must be skipped: the model attribute is
    not updated and they do not appear in the returned bits dict."""
    weights = _make_loaded_weights("vae", "transformer", stored_q=None)
    wd = _build_weight_definition("vae", "transformer")
    models = {"vae": TinyModel(), "transformer": TinyModel()}
    result = WeightApplier.apply_and_quantize(
        weights=weights,
        models=models,
        quantize_arg={"transformer": 8},  # vae is missing
        weight_definition=wd,
    )
    # The transformer was the only one we asked for.
    assert result == {"transformer": 8}


def test_dict_quantize_arg_with_bf16_returns_none_for_that_component():
    """A dict value of None means 'bf16' for that component; the per-component
    result maps it to None."""
    weights = _make_loaded_weights("vae", stored_q=None)
    wd = _build_weight_definition("vae")
    result = WeightApplier.apply_and_quantize(
        weights=weights,
        models={"vae": TinyModel()},
        quantize_arg={"vae": None},  # bf16 for the vae
        weight_definition=wd,
    )
    assert result == {"vae": None}
