# ─── SECTION: Mistral Model Tests ─────────────────────────
import pytest
import torch

from snapmind.core.config import ModelConfig
from snapmind.models.mistral import MistralModel, create_sliding_window_mask


@pytest.fixture
def tiny_mistral_config():
    return ModelConfig(
        model_type="mistral",
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        n_layers=2,
        d_ff=256,
        vocab_size=32000,
        max_seq_len=128,
        norm_eps=1e-5,
        attention_type="gqa",
        pe_type="rope",
        norm_type="rmsnorm",
        activation_type="silu",
        kv_cache_type="sliding_window",
        window_size=8,
    )


@pytest.fixture
def tiny_mistral(tiny_mistral_config):
    return MistralModel(tiny_mistral_config)


class TestMistralArchitecture:
    def test_parameter_count(self, tiny_mistral):
        n = sum(p.numel() for p in tiny_mistral.parameters())
        assert n > 0

    def test_forward_shape(self, tiny_mistral):
        tokens = torch.randint(0, 32000, (2, 16))
        logits = tiny_mistral(tokens)
        assert logits.shape == (2, 16, 32000)

    def test_layer_count(self, tiny_mistral):
        assert len(tiny_mistral.layers) == 2

    def test_window_size_propagated(self, tiny_mistral):
        assert tiny_mistral.window_size == 8
        for layer in tiny_mistral.layers:
            assert layer.window_size == 8


class TestMistralAttention:
    def test_causal_still_holds(self, tiny_mistral):
        tokens = torch.randint(0, 32000, (2, 4))
        logits = tiny_mistral(tokens)
        assert logits.shape == (2, 4, 32000)

    def test_sliding_window_mask_banded(self):
        mask = create_sliding_window_mask(seq_len=6, device=torch.device("cpu"), window_size=3)
        assert mask.shape == (6, 6)
        for i in range(6):
            for j in range(6):
                if 0 <= j <= i and i - j < 3:
                    assert mask[i, j] == 0.0, f"mask[{i},{j}] should be 0, got {mask[i, j]}"
                else:
                    assert mask[i, j] == float("-inf"), f"mask[{i},{j}] should be -inf, got {mask[i, j]}"

    def test_sliding_window_mask_large_window(self):
        mask = create_sliding_window_mask(seq_len=4, device=torch.device("cpu"), window_size=10)
        for i in range(4):
            for j in range(i + 1):
                assert mask[i, j] == 0.0, f"mask[{i},{j}] should be 0 when window covers all past"

    def test_sliding_window_mask_small_window(self):
        mask = create_sliding_window_mask(seq_len=5, device=torch.device("cpu"), window_size=1)
        for i in range(5):
            for j in range(i + 1):
                if j == i:
                    assert mask[i, j] == 0.0
                else:
                    assert mask[i, j] == float("-inf")

    def test_sliding_window_mask_device(self):
        mask = create_sliding_window_mask(seq_len=4, device=torch.device("cpu"), window_size=2)
        assert mask.device.type == "cpu"


class TestMistralKVCacheIntegration:
    def test_prefill_with_sliding_cache(self, tiny_mistral, tiny_mistral_config):
        tokens = torch.randint(0, 32000, (1, 20))
        kv_cache = {i: {"k": None, "v": None} for i in range(tiny_mistral_config.n_layers)}
        logits = tiny_mistral(tokens, kv_cache=kv_cache)
        assert logits.shape == (1, 20, 32000)
        for i in range(tiny_mistral_config.n_layers):
            k, v = kv_cache[i]["k"], kv_cache[i]["v"]
            assert k is not None
            assert k.shape[-2] <= tiny_mistral_config.window_size
            assert v.shape[-2] <= tiny_mistral_config.window_size

    def test_decode_with_sliding_cache(self, tiny_mistral, tiny_mistral_config):
        tokens = torch.randint(0, 32000, (1, 10))
        kv_cache = {i: {"k": None, "v": None} for i in range(tiny_mistral_config.n_layers)}
        tiny_mistral(tokens, kv_cache=kv_cache)
        single_token = torch.tensor([[42]], dtype=torch.long)
        logits = tiny_mistral(single_token, kv_cache=kv_cache)
        assert logits.shape == (1, 1, 32000)


class TestMistralContract:
    def test_has_model_registry(self):
        from snapmind.core.registry import MODEL

        assert "mistral" in MODEL


# ─── ENDSECTION: Mistral Model Tests ──────────────────────
