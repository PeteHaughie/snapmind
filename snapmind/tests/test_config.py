import pytest


class TestModelConfig:
    def test_default_values(self):
        from snapmind.core.config import ModelConfig
        config = ModelConfig()
        assert config.d_model == 4096
        assert config.n_heads == 32
        assert config.n_layers == 32
        assert config.attention_type == "sdpa"
        assert config.pe_type == "rope"

    def test_custom_values(self):
        from snapmind.core.config import ModelConfig
        config = ModelConfig(
            d_model=768,
            n_heads=12,
            n_layers=12,
            vocab_size=50257,
            attention_type="sdpa",
            pe_type="learned",
        )
        assert config.d_model == 768
        assert config.n_heads == 12
        assert config.n_layers == 12

    def test_d_ff_inference(self):
        from snapmind.core.config import ModelConfig
        config = ModelConfig(d_model=768, d_ff=None)
        assert config.d_ff == 3072

    def test_n_kv_heads_defaults_to_n_heads(self):
        from snapmind.core.config import ModelConfig
        config = ModelConfig(n_heads=12, n_kv_heads=None)
        assert config.n_kv_heads == 12

    def test_n_kv_heads_explicit(self):
        from snapmind.core.config import ModelConfig
        config = ModelConfig(n_heads=32, n_kv_heads=8)
        assert config.n_kv_heads == 8

    def test_to_dict(self):
        from snapmind.core.config import ModelConfig
        config = ModelConfig(d_model=768)
        d = config.to_dict()
        assert d["d_model"] == 768
        assert d["model_type"] == "llama"

    def test_from_dict(self):
        from snapmind.core.config import ModelConfig
        d = {"model_type": "gpt2", "d_model": 768, "n_heads": 12}
        config = ModelConfig.from_dict(d)
        assert config.model_type == "gpt2"
        assert config.d_model == 768
        assert config.n_heads == 12

    def test_from_dict_with_defaults(self):
        from snapmind.core.config import ModelConfig
        d = {"model_type": "gpt2"}
        config = ModelConfig.from_dict(d)
        assert config.model_type == "gpt2"
        assert config.d_model == 4096

    def test_round_trip(self):
        from snapmind.core.config import ModelConfig
        original = ModelConfig(d_model=768, n_heads=12, n_layers=12)
        restored = ModelConfig.from_dict(original.to_dict())
        assert restored == original


class TestKVCacheConfig:
    def test_default_values(self):
        from snapmind.core.config import KVCacheConfig
        config = KVCacheConfig()
        assert config.max_seq_len == 8192
        assert config.dtype == "bfloat16"


class TestEngineConfig:
    def test_default_values(self):
        from snapmind.core.config import EngineConfig
        config = EngineConfig()
        assert config.max_batch_tokens == 4096
        assert config.device == "auto"
