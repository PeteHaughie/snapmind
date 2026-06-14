# ─── SECTION: LayerNorm ──────────────────────────────────
import torch
import torch.nn as nn

from snapmind.core.registry import NORM
from snapmind.layers.normalization.base import NormABC


# ANCHOR: LayerNorm
@NORM.register("layernorm")
class LayerNorm(NormABC):
    def __init__(self, normalized_shape: int, eps: float = 1e-5, elementwise_affine: bool = True):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape)) if elementwise_affine else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        if self.bias is not None:
            x_norm = x_norm * self.weight + self.bias
        else:
            x_norm = x_norm * self.weight
        return x_norm


# ENDANCHOR: LayerNorm
# ─── ENDSECTION: LayerNorm ───────────────────────────────
