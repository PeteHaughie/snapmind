# ─── SECTION: RMSNorm ────────────────────────────────────
import torch
import torch.nn as nn

from snapmind.core.registry import NORM
from snapmind.layers.normalization.base import NormABC


# ANCHOR: RMSNorm
@NORM.register("rmsnorm")
class RMSNorm(NormABC):
    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.pow(2).mean(-1, keepdim=True)
        rms = torch.rsqrt(var + self.eps)
        return x * rms * self.weight


# ENDANCHOR: RMSNorm
# ─── ENDSECTION: RMSNorm ─────────────────────────────────
