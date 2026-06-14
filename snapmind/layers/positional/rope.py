# ─── SECTION: Rotary Positional Encoding ─────────────────
import torch
import torch.nn as nn
from snapmind.core.registry import PE
from snapmind.layers.positional.base import PositionalEncodingABC


# ANCHOR: RotaryPositionalEncoding
@PE.register("rope")
class RotaryPositionalEncoding(PositionalEncodingABC):
    def __init__(self, dim: int, max_seq_len: int = 8192, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        self._injection_point = "attention"
        self._register_buffer()

    def _register_buffer(self):
        freqs = 1.0 / (self.theta ** (torch.arange(0, self.dim, 2).float() / self.dim))
        positions = torch.arange(self.max_seq_len)
        angles = torch.outer(positions, freqs)
        cos = angles.cos()
        sin = angles.sin()
        emb = torch.cat([cos, cos], dim=-1)
        self.register_buffer("cos_cached", emb)
        emb_sin = torch.cat([sin, sin], dim=-1)
        self.register_buffer("sin_cached", emb_sin)

    @property
    def injection_point(self):
        return self._injection_point

    def forward(self, x, position_ids=None):
        return x

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def apply_to_qk(self, q: torch.Tensor, k: torch.Tensor, position_ids: torch.Tensor | None = None):
        if position_ids is None:
            seq_len = q.shape[-2]
            cos = self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(0)
            sin = self.sin_cached[:seq_len].unsqueeze(0).unsqueeze(0)
        else:
            cos = self.cos_cached[position_ids].unsqueeze(1)
            sin = self.sin_cached[position_ids].unsqueeze(1)
        q_rot = q * cos + self._rotate_half(q) * sin
        k_rot = k * cos + self._rotate_half(k) * sin
        return q_rot, k_rot
# ENDANCHOR: RotaryPositionalEncoding
# ─── ENDSECTION: Rotary Positional Encoding ──────────────
