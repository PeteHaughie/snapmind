import pytest


class TestWeightLoaderABC:
    def test_cannot_instantiate_directly(self):
        from snapmind.loaders.base import WeightLoaderABC

        with pytest.raises(TypeError):
            WeightLoaderABC()

    def test_minimal_subclass_works(self):
        from snapmind.core.config import ModelConfig
        from snapmind.loaders.base import WeightLoaderABC

        class MinimalLoader(WeightLoaderABC):
            def load(self, path, model, config):
                return True

        loader = MinimalLoader()
        config = ModelConfig()
        assert loader.load("dummy_path", None, config)

    def test_registered_via_LOADER(self):
        from snapmind.core.registry import LOADER

        assert "safetensors" in LOADER
