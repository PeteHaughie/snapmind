# ─── SECTION: FeedForward ────────────────────────────────
import torch
import torch.nn as nn

from snapmind.layers.activation.gelu import GELU


# ANCHOR: FeedForward
class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, activation_type: str = "gelu", dropout: float = 0.0):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=True)
        self.act = GELU()
        self.down_proj = nn.Linear(d_ff, d_model, bias=True)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(self.act(self.gate_proj(x))))


# ENDANCHOR: FeedForward
# ─── ENDSECTION: FeedForward ─────────────────────────────
