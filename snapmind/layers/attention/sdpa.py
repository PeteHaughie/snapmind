# ─── SECTION: Scaled Dot-Product Attention ──────────────
import torch
import torch.nn as nn
import torch.nn.functional as F

from snapmind.core.registry import ATTENTION
from snapmind.layers.attention.base import AttentionABC


# ANCHOR: create_causal_mask
def create_causal_mask(seq_len: int, device=None) -> torch.Tensor:
    mask = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)
    return mask.to(device) if device is not None else mask


# ENDANCHOR: create_causal_mask


# ANCHOR: ScaledDotProductAttention
@ATTENTION.register("sdpa")
class ScaledDotProductAttention(AttentionABC):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0, bias: bool = True):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout

        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        return x.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, seq_len, _ = x.shape
        return x.transpose(1, 2).reshape(batch, seq_len, self.d_model)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: dict | None = None,
        position_ids: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))

        if kv_cache is not None:
            cached_k, cached_v = kv_cache["k"], kv_cache["v"]
            if cached_k is not None:
                k = torch.cat([cached_k, k], dim=-2)
                v = torch.cat([cached_v, v], dim=-2)
            kv_cache["k"], kv_cache["v"] = k, v

        scale = self.head_dim**-0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale

        if mask is not None:
            attn = attn + mask.to(attn.dtype)

        attn_weights = F.softmax(attn, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        out = torch.matmul(attn_weights, v)
        out = self._merge_heads(out)
        out = self.out_proj(out)
        return out, attn_weights


# ENDANCHOR: ScaledDotProductAttention
# ─── ENDSECTION: Scaled Dot-Product Attention ───────────
