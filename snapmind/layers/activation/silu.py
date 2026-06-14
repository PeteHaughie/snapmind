# ─── SECTION: SiLU ───────────────────────────────────────
import torch

from snapmind.core.registry import ACTIVATION
from snapmind.layers.activation.base import ActivationABC


# ANCHOR: SiLU
@ACTIVATION.register("silu")
class SiLU(ActivationABC):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


# ENDANCHOR: SiLU
# ─── ENDSECTION: SiLU ────────────────────────────────────
