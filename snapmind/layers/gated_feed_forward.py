# ─── SECTION: Gated FeedForward ──────────────────────────
import torch
import torch.nn as nn

from snapmind.layers.activation.silu import SiLU


# ANCHOR: GatedFeedForward
class GatedFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
        self.act = SiLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x)))


# ENDANCHOR: GatedFeedForward
# ─── ENDSECTION: Gated FeedForward ───────────────────────
