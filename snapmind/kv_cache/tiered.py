import torch

from snapmind.core.registry import KV_CACHE
from snapmind.kv_cache.base import KVCacheABC


@KV_CACHE.register("tiered")
class TieredPagedKVCache(KVCacheABC):
    def __init__(
        self,
        max_seq_len: int,
        n_layers: int,
        n_heads: int,
        head_dim: int,
        pool_tokens: int = 4096,
        chunk_size: int = 64,
        device: str = "cpu",
    ):
        self.max_seq_len = max_seq_len
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.chunk_size = chunk_size
        self.max_hot_chunks = max(1, pool_tokens // chunk_size)

        self._device = device

        self._chunks: dict[int, list[dict[str, torch.Tensor | None]]] = {}
        self._hot_set: set[int] = set()

        self._active_k: list[torch.Tensor | None] = [None] * n_layers
        self._active_v: list[torch.Tensor | None] = [None] * n_layers
        self._active_chunk_id: int = -1

        self._total_tokens: int = 0
        self._live_dicts: dict[int, dict[str, torch.Tensor | None]] | None = None

    def _assign_device(self) -> str:
        if self._device != "auto":
            return self._device
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _chunk_id(self, seq_pos: int) -> int:
        return seq_pos // self.chunk_size

    def store(self, layer_idx: int, key: torch.Tensor, value: torch.Tensor, seq_pos: int) -> None:
        n_tokens = key.shape[-2]
        for offset in range(n_tokens):
            pos = seq_pos + offset
            cid = self._chunk_id(pos)

            if cid != self._active_chunk_id:
                self._finalize_active_chunk()
                self._active_chunk_id = cid
                self._active_k = [None] * self.n_layers
                self._active_v = [None] * self.n_layers

            k_slice = key[..., offset : offset + 1, :]
            v_slice = value[..., offset : offset + 1, :]
            existing_k = self._active_k[layer_idx]
            existing_v = self._active_v[layer_idx]
            if existing_k is None or existing_v is None:
                self._active_k[layer_idx] = k_slice
                self._active_v[layer_idx] = v_slice
            else:
                self._active_k[layer_idx] = torch.cat([existing_k, k_slice], dim=-2)
                self._active_v[layer_idx] = torch.cat([existing_v, v_slice], dim=-2)

        self._total_tokens = max(self._total_tokens, seq_pos + n_tokens)

    def _finalize_active_chunk(self) -> None:
        if self._active_chunk_id < 0:
            return
        cid = self._active_chunk_id
        device = self._assign_device()
        layer_entries: list[dict[str, torch.Tensor | None]] = []
        all_empty = True
        for i in range(self.n_layers):
            k = self._active_k[i]
            v = self._active_v[i]
            if k is not None and v is not None:
                all_empty = False
                layer_entries.append({"k": k.to(device=device), "v": v.to(device=device)})
            else:
                layer_entries.append({"k": None, "v": None})
        if not all_empty:
            self._chunks[cid] = layer_entries
            self._hot_set.add(cid)
            self._enforce_pool_limit()

    def _enforce_pool_limit(self) -> None:
        if len(self._hot_set) <= self.max_hot_chunks:
            return
        sink = 0
        tail_start = max(0, self._total_tokens - 2 * self.chunk_size)
        pinned = {sink}
        pinned.update(cid for cid in self._hot_set if cid * self.chunk_size >= tail_start)
        evictable = sorted(self._hot_set - pinned, key=lambda c: -c)
        while len(self._hot_set) > self.max_hot_chunks and evictable:
            cid = evictable.pop(0)
            self._hot_set.discard(cid)
            self._offload_chunk(cid)

    def _offload_chunk(self, cid: int) -> None:
        if cid not in self._chunks:
            return
        cpu_device = torch.device("cpu")
        for entry in self._chunks[cid]:
            ek = entry["k"]
            if ek is not None:
                entry["k"] = ek.to(cpu_device)
                ev = entry["v"]
                if ev is not None:
                    entry["v"] = ev.to(cpu_device)

    def score_and_repage(
        self,
        scores: dict[int, float],
        threshold: float = 0.5,
    ) -> None:
        sink = 0
        tail_start = max(0, self._total_tokens - self.chunk_size)
        always_hot = {sink}
        always_hot.update(cid for cid in self._chunks if cid * self.chunk_size >= tail_start)

        device = self._assign_device()
        desired_hot = set(always_hot)
        for cid, score in scores.items():
            if cid in self._chunks and score >= threshold:
                desired_hot.add(cid)

        for cid in list(self._hot_set):
            if cid not in desired_hot:
                self._hot_set.discard(cid)
                self._offload_chunk(cid)

        for cid in desired_hot:
            if cid not in self._hot_set and cid in self._chunks:
                self._hot_set.add(cid)
                for entry in self._chunks[cid]:
                    ek = entry["k"]
                    if ek is not None:
                        entry["k"] = ek.to(device=device)
                        ev = entry["v"]
                        if ev is not None:
                            entry["v"] = ev.to(device=device)

        self._enforce_pool_limit()
        self._rebuild_live_dicts()

    def _rebuild_live_dicts(self) -> None:
        if self._live_dicts is None:
            return
        device = self._assign_device()
        hot_ids = sorted(self._hot_set)

        for i in range(self.n_layers):
            k_parts: list[torch.Tensor] = []
            v_parts: list[torch.Tensor] = []
            for cid in hot_ids:
                entry = self._chunks.get(cid, [{}] * self.n_layers)[i]
                ek = entry.get("k")
                ev = entry.get("v")
                if ek is not None and ev is not None:
                    k_parts.append(ek)
                    v_parts.append(ev)
            active_k = self._active_k[i]
            active_v = self._active_v[i]
            if active_k is not None and active_v is not None:
                k_parts.append(active_k)
                v_parts.append(active_v)

            if k_parts:
                self._live_dicts[i]["k"] = torch.cat(k_parts, dim=-2).to(device=device)
                self._live_dicts[i]["v"] = torch.cat(v_parts, dim=-2).to(device=device)
            else:
                self._live_dicts[i] = {"k": None, "v": None}

    def fetch(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self._live_dicts is not None and layer_idx in self._live_dicts:
            d = self._live_dicts[layer_idx]
            k = d.get("k")
            v = d.get("v")
            if k is not None and v is not None:
                return k, v
        return torch.empty(0), torch.empty(0)

    def layer_dicts(self) -> dict[int, dict[str, torch.Tensor | None]]:
        result: dict[int, dict[str, torch.Tensor | None]] = {}
        for i in range(self.n_layers):
            result[i] = {"k": None, "v": None}
        self._live_dicts = result
        self._rebuild_live_dicts()
        return result

    def evict(self, tokens_to_keep: int) -> None:
        pass

    def reset(self) -> None:
        self._chunks.clear()
        self._hot_set.clear()
        self._active_k = [None] * self.n_layers
        self._active_v = [None] * self.n_layers
        self._active_chunk_id = -1
        self._total_tokens = 0
        self._live_dicts = None

    def memory_usage(self) -> dict:
        gpu_bytes = 0
        cpu_bytes = 0
        for cid, layers in self._chunks.items():
            for entry in layers:
                k = entry.get("k")
                if k is not None:
                    nbytes = k.numel() * k.element_size()
                    if k.device.type == "cpu":
                        cpu_bytes += nbytes * 2
                    else:
                        gpu_bytes += nbytes * 2
        return {
            "gpu_bytes": gpu_bytes,
            "cpu_bytes": cpu_bytes,
            "hot_chunks": len(self._hot_set),
            "cold_chunks": len(self._chunks) - len(self._hot_set),
            "total_tokens": self._total_tokens,
            "pool_slots": self.max_hot_chunks,
        }
