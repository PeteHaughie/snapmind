import pytest
import torch


class TestMLAABC:
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
        import snapmind.layers.attention.mla  # noqa: F401
        from snapmind.core.registry import ATTENTION

        assert "mla" in ATTENTION


TINY_KWARGS = dict(
    d_model=32,
    n_heads=4,
    kv_lora_rank=8,
    qk_nope_head_dim=4,
    qk_rope_head_dim=4,
    v_head_dim=4,
)


class TestMultiHeadLatentAttentionShape:
    @pytest.fixture
    def mla(self):
        from snapmind.layers.attention.mla import MultiHeadLatentAttention

        return MultiHeadLatentAttention(**TINY_KWARGS, max_seq_len=64)

    def test_output_shape(self, mla):
        x = torch.randn(2, 8, 32)
        out, _ = mla(x)
        assert out.shape == (2, 8, 32)

    def test_batch_independent(self, mla):
        x = torch.randn(4, 8, 32)
        out, _ = mla(x)
        assert out.shape == (4, 8, 32)

    def test_different_sequence_length(self, mla):
        x = torch.randn(2, 16, 32)
        out, _ = mla(x)
        assert out.shape == (2, 16, 32)


class TestMultiHeadLatentAttentionLowRankQ:
    @pytest.fixture
    def mla_lowrank(self):
        from snapmind.layers.attention.mla import MultiHeadLatentAttention

        return MultiHeadLatentAttention(**TINY_KWARGS, q_lora_rank=8, max_seq_len=64)

    def test_output_shape(self, mla_lowrank):
        x = torch.randn(2, 8, 32)
        out, _ = mla_lowrank(x)
        assert out.shape == (2, 8, 32)

    def test_has_down_and_up_projections(self, mla_lowrank):
        assert hasattr(mla_lowrank, "q_down_proj")
        assert hasattr(mla_lowrank, "q_up_proj")
        assert not hasattr(mla_lowrank, "q_proj")


class TestMultiHeadLatentAttentionProperties:
    @pytest.fixture
    def mla(self):
        from snapmind.layers.attention.mla import MultiHeadLatentAttention

        return MultiHeadLatentAttention(**TINY_KWARGS, max_seq_len=64)

    @pytest.fixture
    def test_tensor(self):
        return torch.randn(2, 8, 32)

    def test_attention_weights_sum_to_one(self, mla, test_tensor):
        _, weights = mla(test_tensor)
        row_sums = weights.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

    def test_attention_weights_are_nonnegative(self, mla, test_tensor):
        _, weights = mla(test_tensor)
        assert (weights >= 0).all()

    def test_kv_projection_shapes(self, mla):
        assert mla.kv_proj.out_features == 8 + 4
        assert mla.k_up_proj.out_features == 4 * 4
        assert mla.v_up_proj.out_features == 4 * 4

    def test_q_projection_shape(self, mla):
        assert mla.q_proj.out_features == 4 * (4 + 4)

    def test_out_projection_shape(self, mla):
        assert mla.out_proj.out_features == 32

    def test_causal_mask_prevents_future(self, mla, test_tensor):
        from snapmind.layers.attention.sdpa import create_causal_mask

        mask = create_causal_mask(8)
        _, weights = mla(test_tensor, mask=mask)
        upper = torch.triu(weights, diagonal=1)
        assert torch.allclose(upper, torch.zeros_like(upper), atol=1e-6)

    def test_causal_mask_allows_present_and_past(self, mla, test_tensor):
        from snapmind.layers.attention.sdpa import create_causal_mask

        mask = create_causal_mask(8)
        _, weights = mla(test_tensor, mask=mask)
        lower = torch.tril(weights)
        assert lower.sum() > 0

    def test_different_inputs_produce_different_outputs(self, mla):
        x1 = torch.randn(1, 4, 32)
        x2 = torch.randn(1, 4, 32)
        out1, _ = mla(x1)
        out2, _ = mla(x2)
        assert not torch.allclose(out1, out2, atol=1e-4)


class TestMultiHeadLatentAttentionKVCache:
    @pytest.fixture
    def mla(self):
        from snapmind.layers.attention.mla import MultiHeadLatentAttention

        return MultiHeadLatentAttention(**TINY_KWARGS, max_seq_len=64)

    def test_with_cache_extends_sequence(self, mla):
        x1 = torch.randn(1, 4, 32)
        x2 = torch.randn(1, 4, 32)
        kv_cache: dict = {}
        _, _ = mla(x1, kv_cache=kv_cache)
        assert kv_cache["k"].shape[-2] == 4
        _, _ = mla(x2, kv_cache=kv_cache)
        assert kv_cache["k"].shape[-2] == 8

    def test_cache_stores_compressed_latent(self, mla):
        x = torch.randn(1, 4, 32)
        kv_cache: dict = {}
        _, _ = mla(x, kv_cache=kv_cache)
        assert kv_cache["k"].shape[-1] == 8
        assert kv_cache["k_rope"].shape[-1] == 4

    def test_no_bias_by_default(self, mla):
        assert mla.kv_proj.bias is None
        assert mla.k_up_proj.bias is None
        assert mla.v_up_proj.bias is None
        assert mla.q_proj.bias is None
        assert mla.out_proj.bias is None

    def test_kv_cache_improves_memory_for_long_sequences(self, mla):
        from snapmind.layers.attention.mla import MultiHeadLatentAttention

        full = MultiHeadLatentAttention(**TINY_KWARGS, max_seq_len=64)
        chunks = MultiHeadLatentAttention(**TINY_KWARGS, max_seq_len=64)
        x = torch.randn(1, 16, 32)
        out_full, _ = full(x)
        kv_cache: dict = {}
        chunk1 = torch.randn(1, 8, 32)
        chunk2 = torch.randn(1, 8, 32)
        out_c1, _ = chunks(chunk1, kv_cache=kv_cache)
        out_c2, _ = chunks(chunk2, kv_cache=kv_cache)
        assert out_full.shape == (1, 16, 32)
        assert kv_cache["k"].shape[-2] == 16


class TestMultiHeadLatentAttentionFullGQACompatibility:
    def test_mla_can_replace_gqa_tiny_config(self):
        from snapmind.layers.attention.mla import MultiHeadLatentAttention

        mla = MultiHeadLatentAttention(
            d_model=32,
            n_heads=4,
            kv_lora_rank=16,
            qk_nope_head_dim=6,
            qk_rope_head_dim=2,
            v_head_dim=8,
            max_seq_len=64,
        )
        x = torch.randn(2, 8, 32)
        out, weights = mla(x)
        assert out.shape == (2, 8, 32)
        assert weights.shape[-1] == 8
