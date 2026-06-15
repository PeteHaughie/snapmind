# ─── SECTION: Attention ABC ──────────────────────────────
import abc

import torch
import torch.nn as nn


# ANCHOR: AttentionABC
class AttentionABC(nn.Module, abc.ABC):
    """Base class for all attention mechanisms (SDPA, GQA, MLA, …).

    Subclasses must implement :meth:`forward`, which returns ``(output, attn_weights)``.
    """

    @abc.abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        kv_cache: dict | None = None,
        position_ids: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...


# ENDANCHOR: AttentionABC
# ─── ENDSECTION: Attention ABC ───────────────────────────
