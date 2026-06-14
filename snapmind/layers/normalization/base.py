# ─── SECTION: Norm ABC ──────────────────────────────────
import abc

import torch
import torch.nn as nn


# ANCHOR: NormABC
class NormABC(nn.Module, abc.ABC):
    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...


# ENDANCHOR: NormABC
# ─── ENDSECTION: Norm ABC ────────────────────────────────
