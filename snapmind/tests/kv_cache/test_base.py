import pytest


class TestKVCacheABC:
    def test_cannot_instantiate_directly(self):
        from snapmind.kv_cache.base import KVCacheABC

        with pytest.raises(TypeError):
            KVCacheABC()

    def test_minimal_subclass_works(self):
        from snapmind.kv_cache.base import KVCacheABC

        class MinimalCache(KVCacheABC):
            def store(self, layer_idx, key, value, seq_pos):
                pass

            def fetch(self, layer_idx):
                return None, None

            def evict(self, tokens_to_keep):
                pass

            def reset(self):
                pass

            def memory_usage(self):
                return {}

        cache = MinimalCache()
        cache.store(0, None, None, 0)
        k, v = cache.fetch(0)
        assert k is None and v is None

    def test_registered_via_KV_CACHE(self):
        from snapmind.core.registry import KV_CACHE

        assert "naive" in KV_CACHE
