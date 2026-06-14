# ─── SECTION: Attention ABC ──────────────────────────────
import abc
import torch.nn as nn


# ANCHOR: AttentionABC
class AttentionABC(nn.Module, abc.ABC):
    @abc.abstractmethod
    def forward(self, x, kv_cache=None, position_ids=None, mask=None):
        ...
# ENDANCHOR: AttentionABC
# ─── ENDSECTION: Attention ABC ───────────────────────────
