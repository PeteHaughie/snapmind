# ─── SECTION: KV Cache ABC ──────────────────────────────
import abc

import torch


# ANCHOR: KVCacheABC
class KVCacheABC(abc.ABC):
    @abc.abstractmethod
    def store(self, layer_idx: int, key: torch.Tensor, value: torch.Tensor, seq_pos: int) -> None: ...

    @abc.abstractmethod
    def fetch(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]: ...

    @abc.abstractmethod
    def evict(self, tokens_to_keep: int) -> None: ...

    @abc.abstractmethod
    def reset(self) -> None: ...

    @abc.abstractmethod
    def memory_usage(self) -> dict: ...


# ENDANCHOR: KVCacheABC
# ─── ENDSECTION: KV Cache ABC ───────────────────────────
