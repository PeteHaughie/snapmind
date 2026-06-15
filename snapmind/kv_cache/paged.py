# ─── SECTION: Paged KV Cache ────────────────────────────
import torch

from snapmind.core.registry import KV_CACHE
from snapmind.kv_cache.base import KVCacheABC


# ANCHOR: PagedKVCache
@KV_CACHE.register("paged")
class PagedKVCache(KVCacheABC):
    def __init__(
        self,
        max_seq_len: int,
        n_layers: int,
        n_heads: int,
        head_dim: int,
        block_size: int = 16,
        max_blocks: int | None = None,
    ):
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.block_size = block_size
        max_blocks = max_blocks or (max_seq_len + block_size - 1) // block_size
        self.max_blocks = max_blocks

        k_shape = (max_blocks, block_size, n_heads, head_dim)
        self._k_buffer = torch.zeros(k_shape)
        self._v_buffer = torch.zeros(k_shape)
        self._free_blocks = set(range(max_blocks))
        self._block_tables: list[list[int]] = [[] for _ in range(n_layers)]
        self._slot_counts: list[int] = [0] * n_layers

    def store(self, layer_idx: int, key: torch.Tensor, value: torch.Tensor, seq_pos: int) -> None:
        num_tokens = key.shape[-2]
        slot_start = self._slot_counts[layer_idx]

        for offset in range(num_tokens):
            slot = slot_start + offset
            block_idx = slot // self.block_size
            within_block = slot % self.block_size
            bt = self._block_tables[layer_idx]

            while len(bt) <= block_idx:
                if not self._free_blocks:
                    raise RuntimeError("PagedKVCache: out of blocks")
                new_block = self._free_blocks.pop()
                bt.append(new_block)

            phys = bt[block_idx]
            self._k_buffer[phys, within_block] = key[0, :, offset, :]
            self._v_buffer[phys, within_block] = value[0, :, offset, :]

        self._slot_counts[layer_idx] = slot_start + num_tokens

    def fetch(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        bt = self._block_tables[layer_idx]
        if not bt:
            return (
                torch.empty(0, self.n_heads, self.head_dim),
                torch.empty(0, self.n_heads, self.head_dim),
            )
        num_slots = self._slot_counts[layer_idx]
        blocks_needed = (num_slots + self.block_size - 1) // self.block_size
        used_blocks = bt[:blocks_needed]

        k_parts = []
        v_parts = []
        remaining = num_slots
        for b in used_blocks:
            take = min(remaining, self.block_size)
            k_parts.append(self._k_buffer[b, :take])
            v_parts.append(self._v_buffer[b, :take])
            remaining -= take

        k = torch.cat(k_parts, dim=0).permute(1, 0, 2).unsqueeze(0)
        v = torch.cat(v_parts, dim=0).permute(1, 0, 2).unsqueeze(0)
        return k, v

    def evict(self, tokens_to_keep: int) -> None:
        for layer_idx in range(self.n_layers):
            total = self._slot_counts[layer_idx]
            if total <= tokens_to_keep:
                continue

            start_slot = total - tokens_to_keep
            start_block = start_slot // self.block_size
            offset_in_block = start_slot % self.block_size
            bt = self._block_tables[layer_idx]
            blocks_needed = (offset_in_block + tokens_to_keep + self.block_size - 1) // self.block_size
            kept_blocks = bt[start_block:start_block + blocks_needed]
            freed_blocks = bt[:start_block] + bt[start_block + blocks_needed:]
            for b in freed_blocks:
                self._free_blocks.add(b)

            if offset_in_block > 0 and kept_blocks:
                first = kept_blocks[0]
                remaining = tokens_to_keep
                src_off = offset_in_block
                dst_off = 0
                for bi, b in enumerate(kept_blocks):
                    take = min(self.block_size - src_off, remaining)
                    src = self._k_buffer[b, src_off:src_off + take].clone()
                    self._k_buffer[first, dst_off:dst_off + take] = src
                    src = self._v_buffer[b, src_off:src_off + take].clone()
                    self._v_buffer[first, dst_off:dst_off + take] = src
                    dst_off += take
                    remaining -= take
                    src_off = 0
                needed = (dst_off + self.block_size - 1) // self.block_size
                for b in kept_blocks[needed:]:
                    self._free_blocks.add(b)
                kept_blocks = kept_blocks[:needed]

            self._block_tables[layer_idx] = kept_blocks
            self._slot_counts[layer_idx] = tokens_to_keep

    def reset(self) -> None:
        self._k_buffer.zero_()
        self._v_buffer.zero_()
        self._free_blocks = set(range(self.max_blocks))
        self._block_tables = [[] for _ in range(self.n_layers)]
        self._slot_counts = [0] * self.n_layers

    def memory_usage(self) -> dict:
        total_bytes = self._k_buffer.numel() * self._k_buffer.element_size()
        total_bytes += self._v_buffer.numel() * self._v_buffer.element_size()
        total_tokens = sum(self._slot_counts)
        used_blocks = sum(len(bt) for bt in self._block_tables)
        return {
            "num_tokens": total_tokens,
            "total_bytes": total_bytes,
            "used_blocks": used_blocks,
            "max_blocks": self.max_blocks,
        }

    def layer_dicts(self) -> dict[int, dict[str, torch.Tensor | None]]:
        result: dict[int, dict[str, torch.Tensor | None]] = {}
        for i in range(self.n_layers):
            k, v = self.fetch(i)
            if k.numel() == 0:
                result[i] = {"k": None, "v": None}
            else:
                result[i] = {"k": k, "v": v}
        return result


# ENDANCHOR: PagedKVCache
# ─── ENDSECTION: Paged KV Cache ──────────────────────────
