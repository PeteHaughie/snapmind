# ─── SECTION: Learned / No Positional Encoding ──────────
import torch
import torch.nn as nn

from snapmind.core.registry import PE
from snapmind.layers.positional.base import PositionalEncodingABC


# ANCHOR: LearnedPositionalEncoding
@PE.register("learned")
class LearnedPositionalEncoding(PositionalEncodingABC):
    def __init__(self, d_model: int, max_seq_len: int = 1024, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Embedding(max_seq_len, d_model)
        self._injection_point = "embedding"

    @property
    def injection_point(self) -> str:
        return self._injection_point

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor | None = None) -> torch.Tensor:
        seq_len = x.shape[1]
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=x.device).unsqueeze(0)
        pe = self.pe(position_ids)
        return self.dropout(x + pe)


# ENDANCHOR: LearnedPositionalEncoding


# ANCHOR: NoPositionalEncoding
@PE.register("none")
class NoPositionalEncoding(PositionalEncodingABC):
    @property
    def injection_point(self) -> str:
        return "embedding"

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor | None = None) -> torch.Tensor:
        return x


# ENDANCHOR: NoPositionalEncoding
# ─── ENDSECTION: Learned / No Positional Encoding ────────
