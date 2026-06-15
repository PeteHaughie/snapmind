import torch
import torch.nn as nn
import torch.nn.functional as F

from snapmind.core.registry import ATTENTION
from snapmind.layers.attention.base import AttentionABC


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


@ATTENTION.register("mla")
class MultiHeadLatentAttention(AttentionABC):
    rope_cos: torch.Tensor
    rope_sin: torch.Tensor
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        kv_lora_rank: int = 512,
        qk_nope_head_dim: int = 128,
        qk_rope_head_dim: int = 64,
        v_head_dim: int = 128,
        q_lora_rank: int | None = None,
        dropout: float = 0.0,
        bias: bool = False,
        pe=None,
        rope_theta: float = 10000.0,
        max_seq_len: int = 8192,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.dropout = dropout
        self.pe = pe
        self.rope_theta = rope_theta

        kv_out_dim = kv_lora_rank + qk_rope_head_dim
        self.kv_proj = nn.Linear(d_model, kv_out_dim, bias=bias)
        self.k_up_proj = nn.Linear(kv_lora_rank, n_heads * qk_nope_head_dim, bias=bias)
        self.v_up_proj = nn.Linear(kv_lora_rank, n_heads * v_head_dim, bias=bias)

        q_out_dim = n_heads * (qk_nope_head_dim + qk_rope_head_dim)
        if q_lora_rank is not None:
            self.q_down_proj = nn.Linear(d_model, q_lora_rank, bias=bias)
            self.q_up_proj = nn.Linear(q_lora_rank, q_out_dim, bias=bias)
        else:
            self.q_proj = nn.Linear(d_model, q_out_dim, bias=bias)

        self.out_proj = nn.Linear(n_heads * v_head_dim, d_model, bias=bias)

        self._register_rope(max_seq_len)

    def _register_rope(self, max_seq_len: int) -> None:
        dim = self.qk_rope_head_dim
        if dim < 2:
            return
        freqs = 1.0 / (self.rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        angles = torch.outer(positions, freqs)
        cos = angles.cos()
        sin = angles.sin()
        self.register_buffer("rope_cos", torch.cat([cos, cos], dim=-1))
        self.register_buffer("rope_sin", torch.cat([sin, sin], dim=-1))

    def _apply_rope(self, t: torch.Tensor, position_ids: torch.Tensor | None = None) -> torch.Tensor:
        if position_ids is None:
            cos = self.rope_cos[:t.shape[-2]].to(t.dtype)
            sin = self.rope_sin[:t.shape[-2]].to(t.dtype)
            while cos.ndim < t.ndim:
                cos = cos.unsqueeze(0)
                sin = sin.unsqueeze(0)
        else:
            cos = self.rope_cos[position_ids].to(t.dtype)
            sin = self.rope_sin[position_ids].to(t.dtype)
            if cos.ndim == 3 and t.ndim == 4:
                cos = cos.unsqueeze(-2)
                sin = sin.unsqueeze(-2)
        return t * cos + _rotate_half(t) * sin

    def _split_heads(self, x: torch.Tensor, head_dim: int) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        return x.view(batch, seq_len, self.n_heads, head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, n_heads, seq_len, head_dim = x.shape
        return x.transpose(1, 2).reshape(batch, seq_len, n_heads * head_dim)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: dict | None = None,
        position_ids: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, seq_len, _ = x.shape

        if self.q_lora_rank is not None:
            q = self.q_up_proj(self.q_down_proj(x))
        else:
            q = self.q_proj(x)

        nope_dim = self.n_heads * self.qk_nope_head_dim
        q_nope = q[..., :nope_dim]
        q_rope_raw = q[..., nope_dim:]

        kv = self.kv_proj(x)
        k_latent = kv[..., :self.kv_lora_rank]
        k_rope_raw = kv[..., self.kv_lora_rank:]

        if kv_cache is not None:
            cached_klat = kv_cache.get("k")
            cached_krope = kv_cache.get("k_rope")
            if cached_klat is not None:
                assert cached_krope is not None
                k_latent = torch.cat([cached_klat, k_latent], dim=-2)
                k_rope_raw = torch.cat([cached_krope, k_rope_raw], dim=-2)
            kv_cache["k"], kv_cache["k_rope"] = k_latent, k_rope_raw

        k_nope_flat = self.k_up_proj(k_latent)
        v_flat = self.v_up_proj(k_latent)

        q_nope = self._split_heads(q_nope, self.qk_nope_head_dim)
        q_rope = self._split_heads(q_rope_raw, self.qk_rope_head_dim)
        k_nope = self._split_heads(k_nope_flat, self.qk_nope_head_dim)

        q_rope = self._apply_rope(q_rope, position_ids)
        k_rope = self._apply_rope(k_rope_raw, position_ids)
        k_rope = k_rope.unsqueeze(1).expand(-1, self.n_heads, -1, -1)

        q = torch.cat([q_nope, q_rope], dim=-1)
        k = torch.cat([k_nope, k_rope], dim=-1)

        v = self._split_heads(v_flat, self.v_head_dim)

        scale = self.qk_head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale

        if mask is not None:
            attn = attn + mask.to(attn.dtype)

        attn_weights = F.softmax(attn, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        out = torch.matmul(attn_weights, v)
        out = self._merge_heads(out)
        out = self.out_proj(out)
        return out, attn_weights
