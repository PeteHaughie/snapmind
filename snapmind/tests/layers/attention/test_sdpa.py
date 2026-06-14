import pytest
import torch


class TestScaledDotProductAttentionShape:
    def test_output_shape(self, attention_sdpa, test_tensor):
        output, *_ = attention_sdpa(test_tensor)
        assert output.shape == (2, 16, 32)

    def test_batch_independent(self, attention_sdpa):
        x = torch.randn(4, 16, 32)
        output, *_ = attention_sdpa(x)
        assert output.shape == (4, 16, 32)

    def test_different_sequence_length(self, attention_sdpa):
        x = torch.randn(2, 8, 32)
        output, *_ = attention_sdpa(x)
        assert output.shape == (2, 8, 32)


class TestScaledDotProductAttentionProperties:
    def test_attention_weights_sum_to_one(self, attention_sdpa, test_tensor):
        _, weights = attention_sdpa(test_tensor)
        row_sums = weights.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

    def test_attention_weights_are_nonnegative(self, attention_sdpa, test_tensor):
        _, weights = attention_sdpa(test_tensor)
        assert (weights >= 0).all()

    def test_causal_mask_blocks_future_tokens(self, attention_sdpa, test_tensor, test_causal_mask):
        _, weights = attention_sdpa(test_tensor, mask=test_causal_mask)
        upper_tri = torch.triu(weights, diagonal=1)
        assert torch.allclose(upper_tri, torch.zeros_like(upper_tri), atol=1e-6)

    def test_causal_mask_allows_present_and_past(self, attention_sdpa, test_tensor, test_causal_mask):
        _, weights = attention_sdpa(test_tensor, mask=test_causal_mask)
        lower_tri = torch.tril(weights)
        assert lower_tri.sum() > 0


class TestScaledDotProductAttentionWithKVCache:
    def test_with_cache_extends_sequence(self, attention_sdpa, test_tensor):
        initial_output, _ = attention_sdpa(test_tensor)
        kv_cache = {"k": None, "v": None}
        next_tokens = torch.randn(2, 4, 32)
        output, _ = attention_sdpa(next_tokens, kv_cache=kv_cache)
        assert output.shape[-2] == 4


class TestScaledDotProductAttentionValues:
    def test_identical_queries_have_identical_attention(self, attention_sdpa):
        x = torch.randn(1, 1, 32).expand(1, 8, 32)
        _, weights = attention_sdpa(x)
        assert torch.allclose(weights[0, :, 0, :], weights[0, :, 1, :], atol=1e-6)
