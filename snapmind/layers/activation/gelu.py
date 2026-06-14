# ─── SECTION: GELU ───────────────────────────────────────
import torch
import torch.nn as nn

from snapmind.core.registry import ACTIVATION
from snapmind.layers.activation.base import ActivationABC


# ANCHOR: GELU
@ACTIVATION.register("gelu")
class GELU(ActivationABC):
    def __init__(self):
        super().__init__()
        self._gelu = nn.GELU(approximate="tanh")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._gelu(x)


# ENDANCHOR: GELU
# ─── ENDSECTION: GELU ────────────────────────────────────
