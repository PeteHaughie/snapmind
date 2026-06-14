import pytest
import torch


@pytest.fixture
def tiny_config():
    from snapmind.core.config import ModelConfig

    return ModelConfig(
        model_type="gpt2",
        d_model=32,
        n_heads=4,
        n_kv_heads=4,
        n_layers=2,
        d_ff=128,
        vocab_size=256,
        max_seq_len=64,
        norm_eps=1e-5,
        attention_type="sdpa",
        pe_type="learned",
        norm_type="layernorm",
        activation_type="gelu",
        kv_cache_type="naive",
    )


@pytest.fixture
def gpt2_124m_config():
    from snapmind.core.config import ModelConfig

    return ModelConfig(
        model_type="gpt2",
        d_model=768,
        n_heads=12,
        n_kv_heads=12,
        n_layers=12,
        d_ff=3072,
        vocab_size=50257,
        max_seq_len=1024,
        norm_eps=1e-5,
        attention_type="sdpa",
        pe_type="learned",
        norm_type="layernorm",
        activation_type="gelu",
        kv_cache_type="naive",
    )


@pytest.fixture
def mini_config():
    from snapmind.core.config import ModelConfig

    return ModelConfig(
        model_type="gpt2",
        d_model=32,
        n_heads=4,
        n_kv_heads=4,
        n_layers=2,
        d_ff=128,
        vocab_size=50257,
        max_seq_len=64,
        norm_eps=1e-5,
        attention_type="sdpa",
        pe_type="learned",
        norm_type="layernorm",
        activation_type="gelu",
        kv_cache_type="naive",
    )


@pytest.fixture
def mini_gpt2(mini_config):
    from snapmind.models.gpt2 import GPT2Model

    return GPT2Model(mini_config)


@pytest.fixture
def tiny_gpt2(tiny_config):
    from snapmind.models.gpt2 import GPT2Model

    return GPT2Model(tiny_config)


@pytest.fixture
def test_tokens():
    return torch.randint(0, 256, (2, 16))


@pytest.fixture
def test_tensor():
    return torch.randn(2, 16, 32)


@pytest.fixture
def test_causal_mask():
    from snapmind.layers.attention.sdpa import create_causal_mask

    return create_causal_mask(16)


@pytest.fixture
def attention_sdpa():
    from snapmind.layers.attention.sdpa import ScaledDotProductAttention

    return ScaledDotProductAttention(d_model=32, n_heads=4)


@pytest.fixture
def layer_norm():
    from snapmind.layers.normalization.layer_norm import LayerNorm

    return LayerNorm(normalized_shape=32)


@pytest.fixture
def learned_pe():
    from snapmind.layers.positional.learned import LearnedPositionalEncoding

    return LearnedPositionalEncoding(d_model=32, max_seq_len=64, dropout=0.0)


@pytest.fixture
def tiny_llama_config():
    from snapmind.core.config import ModelConfig

    return ModelConfig(
        model_type="llama",
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        n_layers=2,
        d_ff=256,
        vocab_size=32000,
        max_seq_len=64,
        norm_eps=1e-5,
        attention_type="gqa",
        pe_type="rope",
        norm_type="rmsnorm",
        activation_type="silu",
        kv_cache_type="naive",
    )


@pytest.fixture
def tiny_llama(tiny_llama_config):
    from snapmind.models.llama import LlamaModel

    return LlamaModel(tiny_llama_config)


@pytest.fixture
def test_tokens_small_vocab():
    return torch.randint(0, 32000, (2, 16))


@pytest.fixture
def gelu():
    from snapmind.layers.activation.gelu import GELU

    return GELU()
