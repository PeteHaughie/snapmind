import pytest
import torch


@pytest.fixture
def paged_cache():
    from snapmind.kv_cache.paged import PagedKVCache

    return PagedKVCache(max_seq_len=128, n_layers=2, n_heads=4, head_dim=64, block_size=16)


class TestPagedKVCacheStoreFetch:
    def test_store_fetch_round_trip(self, paged_cache):
        k = torch.randn(1, 4, 10, 64)
        v = torch.randn(1, 4, 10, 64)
        paged_cache.store(layer_idx=0, key=k, value=v, seq_pos=0)
        k_out, v_out = paged_cache.fetch(layer_idx=0)
        assert torch.allclose(k, k_out, atol=1e-6)
        assert torch.allclose(v, v_out, atol=1e-6)

    def test_append_grows_sequence(self, paged_cache):
        k1 = torch.randn(1, 4, 5, 64)
        v1 = torch.randn(1, 4, 5, 64)
        paged_cache.store(layer_idx=0, key=k1, value=v1, seq_pos=0)
        k2 = torch.randn(1, 4, 3, 64)
        v2 = torch.randn(1, 4, 3, 64)
        paged_cache.store(layer_idx=0, key=k2, value=v2, seq_pos=5)
        k_out, v_out = paged_cache.fetch(layer_idx=0)
        assert k_out.shape[-2] == 8
        assert torch.allclose(k_out[:, :, :5, :], k1, atol=1e-6)
        assert torch.allclose(k_out[:, :, 5:8, :], k2, atol=1e-6)

    def test_crosses_block_boundary(self, paged_cache):
        k = torch.randn(1, 4, 20, 64)
        v = torch.randn(1, 4, 20, 64)
        paged_cache.store(layer_idx=0, key=k, value=v, seq_pos=0)
        k_out, v_out = paged_cache.fetch(layer_idx=0)
        assert k_out.shape[-2] == 20
        assert torch.allclose(k, k_out, atol=1e-6)

    def test_different_layers_are_independent(self, paged_cache):
        k0 = torch.randn(1, 4, 5, 64)
        k1 = torch.randn(1, 4, 7, 64)
        paged_cache.store(layer_idx=0, key=k0, value=k0, seq_pos=0)
        paged_cache.store(layer_idx=1, key=k1, value=k1, seq_pos=0)
        k0_out, _ = paged_cache.fetch(0)
        k1_out, _ = paged_cache.fetch(1)
        assert k0_out.shape[-2] == 5
        assert k1_out.shape[-2] == 7


class TestPagedKVCacheEviction:
    def test_evict_reduces_length(self, paged_cache):
        k = torch.randn(1, 4, 20, 64)
        paged_cache.store(layer_idx=0, key=k, value=k, seq_pos=0)
        paged_cache.evict(tokens_to_keep=10)
        k_out, _ = paged_cache.fetch(0)
        assert k_out.shape[-2] <= 10

    def test_evict_keeps_most_recent(self, paged_cache):
        k = torch.randn(1, 4, 20, 64)
        paged_cache.store(layer_idx=0, key=k, value=k, seq_pos=0)
        paged_cache.evict(tokens_to_keep=5)
        k_out, _ = paged_cache.fetch(0)
        assert torch.allclose(k_out[:, :, -5:, :], k[:, :, -5:, :], atol=1e-6)

    def test_evict_all_layers(self, paged_cache):
        for i in range(2):
            k = torch.randn(1, 4, 15, 64)
            paged_cache.store(i, k, k, 0)
        paged_cache.evict(tokens_to_keep=3)
        for i in range(2):
            k_out, _ = paged_cache.fetch(i)
            assert k_out.shape[-2] == 3


class TestPagedKVCacheStateManagement:
    def test_reset_clears_all(self, paged_cache):
        paged_cache.store(0, torch.randn(1, 4, 10, 64), torch.randn(1, 4, 10, 64), 0)
        paged_cache.store(1, torch.randn(1, 4, 10, 64), torch.randn(1, 4, 10, 64), 0)
        paged_cache.reset()
        for i in range(2):
            k_out, _ = paged_cache.fetch(i)
            assert k_out.numel() == 0

    def test_empty_cache_returns_empty(self, paged_cache):
        k, v = paged_cache.fetch(0)
        assert k.numel() == 0

    def test_blocks_are_reused_after_eviction(self, paged_cache):
        k = torch.randn(1, 4, 32, 64)
        paged_cache.store(0, k, k, 0)
        usage_before = paged_cache.memory_usage()
        paged_cache.evict(tokens_to_keep=4)
        paged_cache.store(0, torch.randn(1, 4, 10, 64), torch.randn(1, 4, 10, 64), 0)
        usage_after = paged_cache.memory_usage()
        assert usage_after["used_blocks"] <= usage_before["used_blocks"]


class TestPagedKVCacheObservability:
    def test_memory_usage_returns_dict(self, paged_cache):
        usage = paged_cache.memory_usage()
        assert isinstance(usage, dict)

    def test_memory_usage_keys(self, paged_cache):
        paged_cache.store(0, torch.randn(1, 4, 10, 64), torch.randn(1, 4, 10, 64), 0)
        usage = paged_cache.memory_usage()
        assert "num_tokens" in usage
        assert "total_bytes" in usage
        assert "used_blocks" in usage
        assert "max_blocks" in usage
