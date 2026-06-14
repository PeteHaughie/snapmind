import pytest
import torch


@pytest.fixture
def sliding_cache():
    from snapmind.kv_cache.sliding_window import SlidingWindowKVCache

    return SlidingWindowKVCache(max_seq_len=128, n_layers=2, n_heads=4, head_dim=64, window_size=10)


class TestSlidingWindowKVCacheStoreFetch:
    def test_store_fetch_round_trip(self, sliding_cache):
        k = torch.randn(1, 4, 5, 64)
        v = torch.randn(1, 4, 5, 64)
        sliding_cache.store(layer_idx=0, key=k, value=v, seq_pos=0)
        k_out, v_out = sliding_cache.fetch(layer_idx=0)
        assert torch.equal(k, k_out)
        assert torch.equal(v, v_out)

    def test_enforces_window_size(self, sliding_cache):
        k = torch.randn(1, 4, 20, 64)
        v = torch.randn(1, 4, 20, 64)
        sliding_cache.store(layer_idx=0, key=k, value=v, seq_pos=0)
        k_out, v_out = sliding_cache.fetch(layer_idx=0)
        assert k_out.shape[-2] == 10
        assert v_out.shape[-2] == 10

    def test_keeps_most_recent_within_window(self, sliding_cache):
        k = torch.randn(1, 4, 20, 64)
        v = torch.randn(1, 4, 20, 64)
        sliding_cache.store(layer_idx=0, key=k, value=v, seq_pos=0)
        k_out, _ = sliding_cache.fetch(layer_idx=0)
        assert torch.allclose(k_out, k[:, :, -10:, :])

    def test_appends_within_window(self, sliding_cache):
        k1 = torch.randn(1, 4, 6, 64)
        v1 = torch.randn(1, 4, 6, 64)
        sliding_cache.store(layer_idx=0, key=k1, value=v1, seq_pos=0)
        k2 = torch.randn(1, 4, 6, 64)
        v2 = torch.randn(1, 4, 6, 64)
        sliding_cache.store(layer_idx=0, key=k2, value=v2, seq_pos=6)
        k_out, _ = sliding_cache.fetch(layer_idx=0)
        assert k_out.shape[-2] == 10  # window_size
        assert torch.allclose(k_out[:, :, :4, :], k1[:, :, -4:, :])
        assert torch.allclose(k_out[:, :, 4:, :], k2)

    def test_different_layers_are_independent(self, sliding_cache):
        k0 = torch.randn(1, 4, 15, 64)
        k1 = torch.randn(1, 4, 3, 64)
        sliding_cache.store(layer_idx=0, key=k0, value=k0, seq_pos=0)
        sliding_cache.store(layer_idx=1, key=k1, value=k1, seq_pos=0)
        k0_out, _ = sliding_cache.fetch(0)
        k1_out, _ = sliding_cache.fetch(1)
        assert k0_out.shape[-2] == 10  # trimmed to window
        assert k1_out.shape[-2] == 3  # within window


class TestSlidingWindowKVCacheEviction:
    def test_evict_reduces_length(self, sliding_cache):
        k = torch.randn(1, 4, 8, 64)
        sliding_cache.store(layer_idx=0, key=k, value=k, seq_pos=0)
        sliding_cache.evict(tokens_to_keep=3)
        k_out, _ = sliding_cache.fetch(0)
        assert k_out.shape[-2] == 3

    def test_evict_keeps_most_recent(self, sliding_cache):
        k = torch.randn(1, 4, 8, 64)
        sliding_cache.store(layer_idx=0, key=k, value=k, seq_pos=0)
        sliding_cache.evict(tokens_to_keep=3)
        k_out, _ = sliding_cache.fetch(0)
        assert torch.allclose(k_out, k[:, :, -3:, :])


class TestSlidingWindowKVCacheStateManagement:
    def test_reset_clears_all(self, sliding_cache):
        sliding_cache.store(0, torch.randn(1, 4, 8, 64), torch.randn(1, 4, 8, 64), 0)
        sliding_cache.reset()
        k_out, _ = sliding_cache.fetch(0)
        assert k_out.numel() == 0

    def test_empty_cache_returns_empty(self, sliding_cache):
        k, v = sliding_cache.fetch(0)
        assert k.numel() == 0
        assert v.numel() == 0


class TestSlidingWindowKVCacheObservability:
    def test_memory_usage_returns_dict(self, sliding_cache):
        sliding_cache.store(0, torch.randn(1, 4, 8, 64), torch.randn(1, 4, 8, 64), 0)
        usage = sliding_cache.memory_usage()
        assert isinstance(usage, dict)
        assert "num_tokens" in usage
        assert "total_bytes" in usage
