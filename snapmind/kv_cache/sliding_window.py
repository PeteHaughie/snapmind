# ─── SECTION: Sliding Window KV Cache ─────────────────────
import torch

from snapmind.core.registry import KV_CACHE
from snapmind.kv_cache.base import KVCacheABC


# ANCHOR: SlidingWindowKVCache
@KV_CACHE.register("sliding_window")
class SlidingWindowKVCache(KVCacheABC):
    def __init__(self, max_seq_len: int, n_layers: int, n_heads: int, head_dim: int, window_size: int = 4096):
        self.max_seq_len = max_seq_len
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.window_size = window_size
        self._caches: dict[int, dict[str, torch.Tensor | None]] = {i: {"k": None, "v": None} for i in range(n_layers)}

    def store(self, layer_idx: int, key: torch.Tensor, value: torch.Tensor, seq_pos: int) -> None:
        cache = self._caches[layer_idx]
        cached_k = cache["k"]
        if cached_k is None:
            cache["k"] = key
            cache["v"] = value
        else:
            cache["k"] = torch.cat([cached_k, key], dim=-2)
            cache["v"] = torch.cat([cached_k, value], dim=-2)
        self._trim_to_window(layer_idx)

    def _trim_to_window(self, layer_idx: int) -> None:
        cache = self._caches[layer_idx]
        k = cache["k"]
        v = cache["v"]
        if k is not None and v is not None and k.shape[-2] > self.window_size:
            cache["k"] = k[..., -self.window_size :, :]
            cache["v"] = v[..., -self.window_size :, :]

    def fetch(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        cache = self._caches[layer_idx]
        k = cache["k"]
        v = cache["v"]
        if k is None or v is None:
            return (
                torch.empty(0, dtype=torch.float32),
                torch.empty(0, dtype=torch.float32),
            )
        return k, v

    def evict(self, tokens_to_keep: int) -> None:
        for i in range(self.n_layers):
            cache = self._caches[i]
            k = cache["k"]
            v = cache["v"]
            if k is not None and v is not None:
                cache["k"] = k[..., -tokens_to_keep:, :]
                cache["v"] = v[..., -tokens_to_keep:, :]

    def reset(self) -> None:
        for i in range(self.n_layers):
            self._caches[i] = {"k": None, "v": None}

    def memory_usage(self) -> dict:
        total_bytes = 0
        num_tokens = 0
        for i in range(self.n_layers):
            cache = self._caches[i]
            k = cache["k"]
            if k is not None:
                total_bytes += k.element_size() * k.numel()
                v = cache["v"]
                if v is not None:
                    total_bytes += v.element_size() * v.numel()
                num_tokens += k.shape[-2]
        return {"num_tokens": num_tokens, "total_bytes": total_bytes}

    def layer_dicts(self) -> dict[int, dict[str, torch.Tensor | None]]:
        return self._caches


# ENDANCHOR: SlidingWindowKVCache
# ─── ENDSECTION: Sliding Window KV Cache ──────────────────
