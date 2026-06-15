# ─── SECTION: SafetensorsLoader Tests ─────────────────────
import tempfile

import pytest

from snapmind.core.config import ModelConfig
from snapmind.core.registry import RegistryError
from snapmind.loaders.safetensors import SafetensorsLoader


class TestSafetensorsLoaderLocal:
    def test_load_from_local_path_returns_expected_keys(self, tiny_gpt2, tiny_config):
        state_dict = {k: v.clone() for k, v in tiny_gpt2.state_dict().items()}
        with tempfile.NamedTemporaryFile(suffix=".safetensors") as f:
            from safetensors.torch import save_file

            save_file(state_dict, f.name)
            loader = SafetensorsLoader()
            result = loader.load(f.name, tiny_gpt2, tiny_config)
        assert "missing" in result
        assert "unexpected" in result
        assert "path" in result
        assert result["path"] == f.name

    def test_invalid_path(self, tiny_gpt2, tiny_config):
        loader = SafetensorsLoader()
        with pytest.raises(Exception):
            loader.load("/nonexistent/path/model.safetensors", tiny_gpt2, tiny_config)


class TestSafetensorsLoaderHF:
    def test_raises_for_unknown_model(self, tiny_gpt2):
        unknown_config = ModelConfig(model_type="unknown_model")
        loader = SafetensorsLoader()
        with pytest.raises(RegistryError, match="unknown key 'unknown_model'"):
            loader.load(None, tiny_gpt2, unknown_config)


class TestSafetensorsLoaderContract:
    def test_extends_abc(self):
        from snapmind.loaders.base import WeightLoaderABC

        assert issubclass(SafetensorsLoader, WeightLoaderABC)


# ─── ENDSECTION: SafetensorsLoader Tests ──────────────────
