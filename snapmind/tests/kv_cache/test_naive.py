import pytest
import torch


@pytest.fixture
def naive_cache():
    from snapmind.kv_cache.naive import NaiveKVCache

    return NaiveKVCache(max_seq_len=128, n_layers=2, n_heads=4, head_dim=64)


class TestNaiveKVCacheStoreFetch:
    def test_store_fetch_round_trip(self, naive_cache):
        k = torch.randn(1, 4, 10, 64)
        v = torch.randn(1, 4, 10, 64)
        naive_cache.store(layer_idx=0, key=k, value=v, seq_pos=0)
        k_out, v_out = naive_cache.fetch(layer_idx=0)
        assert torch.equal(k, k_out)
        assert torch.equal(v, v_out)

    def test_append_grows_sequence(self, naive_cache):
        k1 = torch.randn(1, 4, 5, 64)
        v1 = torch.randn(1, 4, 5, 64)
        naive_cache.store(layer_idx=0, key=k1, value=v1, seq_pos=0)
        k2 = torch.randn(1, 4, 3, 64)
        v2 = torch.randn(1, 4, 3, 64)
        naive_cache.store(layer_idx=0, key=k2, value=v2, seq_pos=5)
        k_out, v_out = naive_cache.fetch(layer_idx=0)
        assert k_out.shape[-2] == 8
        assert torch.allclose(k_out[:, :, :5, :], k1)
        assert torch.allclose(k_out[:, :, 5:8, :], k2)

    def test_different_layers_are_independent(self, naive_cache):
        k0 = torch.randn(1, 4, 5, 64)
        k1 = torch.randn(1, 4, 7, 64)
        naive_cache.store(layer_idx=0, key=k0, value=k0, seq_pos=0)
        naive_cache.store(layer_idx=1, key=k1, value=k1, seq_pos=0)
        k0_out, _ = naive_cache.fetch(0)
        k1_out, _ = naive_cache.fetch(1)
        assert k0_out.shape[-2] == 5
        assert k1_out.shape[-2] == 7


class TestNaiveKVCacheEviction:
    def test_evict_reduces_length(self, naive_cache):
        k = torch.randn(1, 4, 20, 64)
        naive_cache.store(layer_idx=0, key=k, value=k, seq_pos=0)
        naive_cache.evict(tokens_to_keep=10)
        k_out, _ = naive_cache.fetch(0)
        assert k_out.shape[-2] <= 10

    def test_evict_keeps_most_recent(self, naive_cache):
        k = torch.randn(1, 4, 20, 64)
        naive_cache.store(layer_idx=0, key=k, value=k, seq_pos=0)
        naive_cache.evict(tokens_to_keep=5)
        k_out, _ = naive_cache.fetch(0)
        assert torch.allclose(k_out[:, :, -5:, :], k[:, :, -5:, :])

    def test_evict_all_layers(self, naive_cache):
        for i in range(2):
            naive_cache.store(i, torch.randn(1, 4, 15, 64), torch.randn(1, 4, 15, 64), 0)
        naive_cache.evict(tokens_to_keep=3)
        for i in range(2):
            k_out, _ = naive_cache.fetch(i)
            assert k_out.shape[-2] == 3


class TestNaiveKVCacheStateManagement:
    def test_reset_clears_all(self, naive_cache):
        naive_cache.store(0, torch.randn(1, 4, 10, 64), torch.randn(1, 4, 10, 64), 0)
        naive_cache.store(1, torch.randn(1, 4, 10, 64), torch.randn(1, 4, 10, 64), 0)
        naive_cache.reset()
        for i in range(2):
            k_out, _ = naive_cache.fetch(i)
            assert k_out.numel() == 0

    def test_empty_cache_returns_empty(self, naive_cache):
        k, v = naive_cache.fetch(0)
        assert k.numel() == 0
        assert v.numel() == 0


class TestNaiveKVCacheObservability:
    def test_memory_usage_returns_dict(self, naive_cache):
        usage = naive_cache.memory_usage()
        assert isinstance(usage, dict)

    def test_memory_usage_keys(self, naive_cache):
        naive_cache.store(0, torch.randn(1, 4, 10, 64), torch.randn(1, 4, 10, 64), 0)
        usage = naive_cache.memory_usage()
        assert "num_tokens" in usage
        assert "total_bytes" in usage
