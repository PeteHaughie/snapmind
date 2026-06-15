# ─── SECTION: KV Cache ABC ──────────────────────────────
import abc
from typing import Any

import torch


# ANCHOR: KVCacheABC
class KVCacheABC(abc.ABC):
    """Base class for KV cache strategies (naive, sliding window, paged, …).

    Subclasses manage per-layer key/value storage and expose standard lifecycle
    operations: :meth:`store`, :meth:`fetch`, :meth:`evict`, :meth:`reset`,
    and :meth:`layer_dicts`.
    """

    @abc.abstractmethod
    def store(self, layer_idx: int, key: torch.Tensor, value: torch.Tensor, seq_pos: int) -> None:
        """Store *key* and *value* tensors for the given layer at *seq_pos*."""

    @abc.abstractmethod
    def fetch(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(keys, values)`` for the given layer, concatenated along the sequence dim."""

    @abc.abstractmethod
    def evict(self, tokens_to_keep: int) -> None:
        """Drop all but the *tokens_to_keep* most recent entries."""

    @abc.abstractmethod
    def reset(self) -> None:
        """Clear all cached data."""

    @abc.abstractmethod
    def memory_usage(self) -> dict:
        """Return a dict with memory/block statistics (keys vary by strategy)."""

    @abc.abstractmethod
    def layer_dicts(self) -> dict[int, dict[str, Any]]:
        """Return the per-layer ``{layer_idx: {"k": tensor, "v": tensor}}`` dict.

        The returned dicts are live references — mutations by attention layers
        (in-place tensor updates) propagate back to the cache.
        """


# ENDANCHOR: KVCacheABC
# ─── ENDSECTION: KV Cache ABC ───────────────────────────
