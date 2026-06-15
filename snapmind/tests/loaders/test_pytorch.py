# ─── SECTION: PyTorchLoader Tests ─────────────────────────
import tempfile

import pytest

from snapmind.core.config import ModelConfig
from snapmind.core.registry import RegistryError
from snapmind.loaders.pytorch import PyTorchLoader


class TestPyTorchLoaderLocal:
    def test_load_from_local_path_returns_expected_keys(self, tiny_gpt2, tiny_config):
        with tempfile.NamedTemporaryFile(suffix=".bin") as f:
            import torch

            torch.save({k: v.clone() for k, v in tiny_gpt2.state_dict().items()}, f.name)
            f.flush()
            loader = PyTorchLoader()
            result = loader.load(f.name, tiny_gpt2, tiny_config)
        assert "missing" in result
        assert "unexpected" in result
        assert "path" in result
        assert result["path"] == f.name

    def test_invalid_path(self, tiny_gpt2, tiny_config):
        loader = PyTorchLoader()
        with pytest.raises(Exception):
            loader.load("/nonexistent/path/model.bin", tiny_gpt2, tiny_config)


class TestPyTorchLoaderHF:
    def test_raises_for_unknown_model(self, tiny_gpt2):
        unknown_config = ModelConfig(model_type="unknown_model")
        loader = PyTorchLoader()
        with pytest.raises(RegistryError, match="unknown key 'unknown_model'"):
            loader.load(None, tiny_gpt2, unknown_config)


class TestPyTorchLoaderContract:
    def test_extends_abc(self):
        from snapmind.loaders.base import WeightLoaderABC

        assert issubclass(PyTorchLoader, WeightLoaderABC)


# ─── ENDSECTION: PyTorchLoader Tests ─────────────────────
