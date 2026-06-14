import pytest
import torch


class TestGQAABC:
    def test_cannot_instantiate_directly(self):
        from snapmind.layers.attention.base import AttentionABC

        with pytest.raises(TypeError):
            AttentionABC()

    def test_minimal_subclass_works(self):
        from snapmind.layers.attention.base import AttentionABC

        class MinimalAttn(AttentionABC):
            def forward(self, x, kv_cache=None, position_ids=None, mask=None):
                return x, None

        m = MinimalAttn()
        t = torch.randn(2, 4, 8)
        out, _ = m(t)
        assert out.shape == (2, 4, 8)

    def test_registered_in_attention_registry(self):
        from snapmind.core.registry import ATTENTION

        assert "gqa" in ATTENTION


class TestGroupedQueryAttention:
    @pytest.fixture
    def gqa(self):
        from snapmind.layers.attention.gqa import GroupedQueryAttention

        return GroupedQueryAttention(d_model=32, n_heads=4, n_kv_heads=2)

    def test_attention_weights_sum_to_one(self, gqa):
        x = torch.randn(2, 8, 32)
        _, weights = gqa(x)
        batch, n_heads, seq, _ = weights.shape
        assert torch.allclose(weights.sum(dim=-1), torch.ones(batch, n_heads, seq), atol=1e-5)

    def test_kv_heads_less_than_q_heads(self, gqa):
        assert gqa.n_kv_heads == 2
        assert gqa.n_heads == 4

    def test_qkv_projection_shapes(self, gqa):
        assert gqa.q_proj.out_features == 4 * 8
        assert gqa.k_proj.out_features == 2 * 8
        assert gqa.v_proj.out_features == 2 * 8

    def test_causal_mask_prevents_future(self, gqa):
        from snapmind.layers.attention.sdpa import create_causal_mask

        x = torch.randn(1, 8, 32)
        mask = create_causal_mask(8)
        _, weights = gqa(x, mask=mask)
        assert (weights.triu(diagonal=1) == 0).all()

    def test_output_shape(self, gqa):
        x = torch.randn(2, 8, 32)
        out, _ = gqa(x)
        assert out.shape == (2, 8, 32)

    def test_kv_cache_updates(self, gqa):
        x1 = torch.randn(1, 4, 32)
        x2 = torch.randn(1, 4, 32)
        kv_cache = {"k": None, "v": None}
        out1, _ = gqa(x1, kv_cache=kv_cache)
        assert kv_cache["k"].shape[-2] == 4
        out2, _ = gqa(x2, kv_cache=kv_cache)
        assert kv_cache["k"].shape[-2] == 8

    def test_no_bias_by_default(self, gqa):
        assert gqa.q_proj.bias is None
        assert gqa.k_proj.bias is None
        assert gqa.v_proj.bias is None
        assert gqa.out_proj.bias is None
