# ─── SECTION: TransformerBlock ──────────────────────────
import torch
import torch.nn as nn

from snapmind.core.config import ModelConfig
from snapmind.layers.attention.sdpa import ScaledDotProductAttention, create_causal_mask
from snapmind.layers.feed_forward import FeedForward
from snapmind.layers.normalization.layer_norm import LayerNorm


# ANCHOR: TransformerBlock
class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        d_ff: int = config.d_ff if config.d_ff is not None else config.d_model * 4
        self.self_attn = ScaledDotProductAttention(
            d_model=config.d_model,
            n_heads=config.n_heads,
            dropout=config.dropout,
        )
        self.ln1 = LayerNorm(normalized_shape=config.d_model, eps=config.norm_eps)
        self.ln2 = LayerNorm(normalized_shape=config.d_model, eps=config.norm_eps)
        self.feed_forward = FeedForward(
            d_model=config.d_model,
            d_ff=d_ff,
            activation_type=config.activation_type,
            dropout=config.dropout,
        )

    def forward(
        self, x: torch.Tensor, kv_cache: dict | None = None, position_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        is_prefill = kv_cache is None or kv_cache.get("k") is None
        mask = create_causal_mask(x.shape[1], x.device) if is_prefill else None
        attn_out, _ = self.self_attn(self.ln1(x), kv_cache=kv_cache, mask=mask)
        x = x + attn_out
        ffn_out = self.feed_forward(self.ln2(x))
        x = x + ffn_out
        return x


# ENDANCHOR: TransformerBlock
# ─── ENDSECTION: TransformerBlock ────────────────────────
