# ─── SECTION: Grouped Query Attention ────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F

from snapmind.core.registry import ATTENTION
from snapmind.layers.attention.base import AttentionABC
from snapmind.layers.positional.base import PositionalEncodingABC


# ANCHOR: GroupedQueryAttention
@ATTENTION.register("gqa")
class GroupedQueryAttention(AttentionABC):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int | None = None,
        dropout: float = 0.0,
        bias: bool = False,
        pe: PositionalEncodingABC | None = None,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout
        self.pe = pe

        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

    def _split_heads(self, x: torch.Tensor, n_heads: int) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        return x.view(batch, seq_len, n_heads, self.head_dim).transpose(1, 2)

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
        q = self._split_heads(self.q_proj(x), self.n_heads)
        k = self._split_heads(self.k_proj(x), self.n_kv_heads)
        v = self._split_heads(self.v_proj(x), self.n_kv_heads)

        if self.pe is not None:
            q, k = self.pe.apply_to_qk(q, k, position_ids)

        if kv_cache is not None:
            cached_k, cached_v = kv_cache["k"], kv_cache["v"]
            if cached_k is not None:
                k = torch.cat([cached_k, k], dim=-2)
                v = torch.cat([cached_v, v], dim=-2)
            kv_cache["k"], kv_cache["v"] = k, v

        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

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


# ENDANCHOR: GroupedQueryAttention
# ─── ENDSECTION: Grouped Query Attention ─────────────────
