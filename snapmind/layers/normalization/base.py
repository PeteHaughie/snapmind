# ─── SECTION: Norm ABC ──────────────────────────────────
import abc

import torch
import torch.nn as nn


# ANCHOR: NormABC
class NormABC(nn.Module, abc.ABC):
    """Base class for normalization layers (LayerNorm, RMSNorm, …).

    Subclasses must implement :meth:`forward` with a single ``(batch, seq, dim)`` input.
    """

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize *x* along the last dimension."""


# ENDANCHOR: NormABC
# ─── ENDSECTION: Norm ABC ────────────────────────────────
