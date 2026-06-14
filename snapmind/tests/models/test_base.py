import pytest
import torch


class TestBaseModelABC:
    def test_cannot_instantiate_directly(self):
        from snapmind.models.base import BaseModelABC

        with pytest.raises(TypeError):
            BaseModelABC()

    def test_minimal_subclass_works(self):
        from snapmind.core.config import ModelConfig
        from snapmind.models.base import BaseModelABC

        class MinimalModel(BaseModelABC):
            def __init__(self, config):
                super().__init__(config)
                self.embed = torch.nn.Embedding(256, 32)

            def forward(self, tokens, kv_cache=None, position_ids=None):
                return self.embed(tokens)

        config = ModelConfig(d_model=32)
        model = MinimalModel(config)
        x = torch.randint(0, 256, (2, 16))
        result = model(x)
        assert result.shape == (2, 16, 32)

    def test_config_is_stored(self):
        from snapmind.core.config import ModelConfig
        from snapmind.models.base import BaseModelABC

        class MinimalModel(BaseModelABC):
            def __init__(self, config):
                super().__init__(config)
                self.embed = torch.nn.Embedding(config.vocab_size, config.d_model)

            def forward(self, tokens, kv_cache=None, position_ids=None):
                return self.embed(tokens)

        config = ModelConfig(d_model=768)
        model = MinimalModel(config)
        assert model.config is config
        assert model.config.d_model == 768
