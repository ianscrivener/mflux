# CLIs hard-wired to one model must honour or reject --model, never silently ignore it.
# Regression tests for the bug where mflux-generate-krea2 --model dev still constructed
# krea/Krea-2-Turbo without a word of warning (same story on the z-image-turbo and both
# ernie CLIs). Each single-model CLI now routes --model through
# ConfigResolution.resolve_restricted: builtin registry names must be an alias of the
# CLI's own model, while paths and HuggingFace repo ids (which parse_args marks by
# setting model_path) keep the CLI's own config and load weights from the path, as they
# always have.

import sys

import pytest

from mflux.models.common.config.model_config import AVAILABLE_MODELS
from mflux.models.common.resolution.config_resolution import ConfigResolution
from mflux.models.ernie_image.cli import ernie_image_generate, ernie_image_turbo_generate
from mflux.models.krea2.cli import krea2_generate
from mflux.models.lens.cli import lens_generate
from mflux.models.z_image.cli import z_image_turbo_generate
from mflux.utils.exceptions import ModelConfigError

# (CLI module, registry key, a foreign model that must be rejected).
CLI_MODELS = [
    (krea2_generate, "krea-2", "dev"),
    (z_image_turbo_generate, "z-image-turbo", "dev"),
    (ernie_image_generate, "ernie-image", "ernie-image-turbo"),
    (ernie_image_turbo_generate, "ernie-image-turbo", "ernie-image"),
    (lens_generate, "lens-turbo", "dev"),
]


@pytest.mark.fast
class TestRestrictedModelConfig:
    # The CLI's own parser derives model_path from --model, so these tests go through
    # build_parser().parse_args() rather than constructing argument combinations no CLI
    # can produce.
    @staticmethod
    def _resolve_via_parser(monkeypatch, module, registry_key, extra_argv=()):
        monkeypatch.setattr(sys, "argv", ["prog", "--prompt", "test", *extra_argv])
        args = module.build_parser().parse_args()
        return ConfigResolution.resolve_restricted(args.model, registry_key, model_path=args.model_path)

    @pytest.mark.parametrize("module,registry_key,foreign", CLI_MODELS, ids=lambda v: getattr(v, "__name__", v))
    def test_omitted_model_returns_registry_entry(self, monkeypatch, module, registry_key, foreign):
        config = self._resolve_via_parser(monkeypatch, module, registry_key)
        assert config is AVAILABLE_MODELS[registry_key]

    @pytest.mark.parametrize("module,registry_key,foreign", CLI_MODELS, ids=lambda v: getattr(v, "__name__", v))
    def test_all_aliases_accepted(self, monkeypatch, module, registry_key, foreign):
        expected = AVAILABLE_MODELS[registry_key]
        for alias in expected.aliases:
            config = self._resolve_via_parser(monkeypatch, module, registry_key, ["--model", alias])
            assert config is expected

    @pytest.mark.parametrize("module,registry_key,foreign", CLI_MODELS, ids=lambda v: getattr(v, "__name__", v))
    def test_foreign_model_rejected(self, monkeypatch, module, registry_key, foreign):
        with pytest.raises(ModelConfigError, match="only accepts the aliases"):
            self._resolve_via_parser(monkeypatch, module, registry_key, ["--model", foreign])

    def test_krea2_raw_rejected_by_krea2_cli(self, monkeypatch):
        # Same architecture, but the generate CLI runs the Turbo checkpoint only; Raw is
        # the training base and must not be silently swapped for Turbo.
        with pytest.raises(ModelConfigError, match="only accepts the aliases"):
            self._resolve_via_parser(monkeypatch, krea2_generate, "krea-2", ["--model", "krea-2-raw"])

    def test_z_image_controlnet_alias_rejected_despite_shared_repo_id(self, monkeypatch):
        # z-image-turbo and its ControlNet share model_name "Tongyi-MAI/Z-Image-Turbo";
        # identity comparison keeps the ControlNet alias out of the plain turbo CLI.
        assert (
            AVAILABLE_MODELS["z-image-turbo"].model_name
            == AVAILABLE_MODELS["z-image-turbo-controlnet-union-2.1"].model_name
        )
        with pytest.raises(ModelConfigError, match="only accepts the aliases"):
            self._resolve_via_parser(
                monkeypatch, z_image_turbo_generate, "z-image-turbo", ["--model", "z-image-controlnet"]
            )

    def test_own_repo_id_keeps_cli_config(self, monkeypatch):
        # The repo id is not a builtin spelling, so parse_args routes it through
        # model_path; validation must not judge it (exact-match on this shared repo id
        # resolves to the ControlNet entry). Metadata reruns (-C) restore the repo id
        # from the sidecar and take this same path.
        config = self._resolve_via_parser(
            monkeypatch, z_image_turbo_generate, "z-image-turbo", ["--model", "Tongyi-MAI/Z-Image-Turbo"]
        )
        assert config is AVAILABLE_MODELS["z-image-turbo"]

    @pytest.mark.parametrize(
        "module,registry_key,path",
        [
            # Directory names whose substrings infer to a different model on main's
            # resolution rules; each loaded on main and must keep loading.
            (z_image_turbo_generate, "z-image-turbo", "~/models/zimage-q8"),
            (krea2_generate, "krea-2", "~/Developer/mflux/my-turbo-q8"),
            (ernie_image_turbo_generate, "ernie-image-turbo", "~/models/ernie-image-q4"),
        ],
        ids=lambda v: getattr(v, "__name__", v),
    )
    def test_local_checkpoint_path_keeps_cli_config(self, monkeypatch, module, registry_key, path):
        config = self._resolve_via_parser(monkeypatch, module, registry_key, ["--model", path])
        assert config is AVAILABLE_MODELS[registry_key]

    def test_saved_checkpoint_name_keeps_cli_config(self, monkeypatch):
        config = self._resolve_via_parser(monkeypatch, krea2_generate, "krea-2", ["--model", "my-krea-2-finetune"])
        assert config is AVAILABLE_MODELS["krea-2"]
