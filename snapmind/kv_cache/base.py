# ─── SECTION: KV Cache ABC ──────────────────────────────
import abc


# ANCHOR: KVCacheABC
class KVCacheABC(abc.ABC):
    @abc.abstractmethod
    def store(self, layer_idx, key, value, seq_pos):
        ...

    @abc.abstractmethod
    def fetch(self, layer_idx):
        ...

    @abc.abstractmethod
    def evict(self, tokens_to_keep):
        ...

    @abc.abstractmethod
    def reset(self):
        ...

    @abc.abstractmethod
    def memory_usage(self):
        ...
# ENDANCHOR: KVCacheABC
# ─── ENDSECTION: KV Cache ABC ───────────────────────────
