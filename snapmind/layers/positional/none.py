# ─── SECTION: No Positional Encoding ─────────────────────
import torch

from snapmind.core.registry import PE
from snapmind.layers.positional.base import PositionalEncodingABC


# ANCHOR: NoPositionalEncoding
@PE.register("none")
class NoPositionalEncoding(PositionalEncodingABC):
    @property
    def injection_point(self) -> str:
        return "embedding"

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor | None = None) -> torch.Tensor:
        return x


# ENDANCHOR: NoPositionalEncoding
# ─── ENDSECTION: No Positional Encoding ──────────────────
