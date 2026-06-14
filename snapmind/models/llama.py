# ─── SECTION: Llama Model ───────────────────────────────
import torch.nn as nn
from snapmind.core.registry import MODEL
from snapmind.core.config import ModelConfig
from snapmind.layers.normalization.rms_norm import RMSNorm
from snapmind.layers.positional.rope import RotaryPositionalEncoding
from snapmind.layers.attention.gqa import GroupedQueryAttention
from snapmind.layers.gated_feed_forward import GatedFeedForward
from snapmind.models.base import BaseModelABC


# ANCHOR: LlamaTransformerBlock
class LlamaTransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int, pe: RotaryPositionalEncoding):
        super().__init__()
        self.layer_idx = layer_idx
        self.self_attn = GroupedQueryAttention(
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_kv_heads=config.n_kv_heads,
            dropout=config.dropout,
            bias=False,
            pe=pe,
        )
        self.input_layernorm = RMSNorm(normalized_shape=config.d_model, eps=config.norm_eps)
        self.post_attention_layernorm = RMSNorm(normalized_shape=config.d_model, eps=config.norm_eps)
        self.mlp = GatedFeedForward(
            d_model=config.d_model,
            d_ff=config.d_ff,
            dropout=config.dropout,
        )

    def forward(self, x, kv_cache=None, position_ids=None):
        is_prefill = kv_cache is None or kv_cache.get("k") is None
        from snapmind.layers.attention.sdpa import create_causal_mask
        mask = create_causal_mask(x.shape[1], x.device) if is_prefill else None
        attn_out, _ = self.self_attn(self.input_layernorm(x), kv_cache=kv_cache, mask=mask, position_ids=position_ids)
        x = x + attn_out
        ffn_out = self.mlp(self.post_attention_layernorm(x))
        x = x + ffn_out
        return x
# ENDANCHOR: LlamaTransformerBlock


# ANCHOR: LlamaModel
@MODEL.register("llama")
class LlamaModel(BaseModelABC):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        head_dim = config.d_model // config.n_heads
        self.pe = RotaryPositionalEncoding(
            dim=head_dim,
            max_seq_len=config.max_seq_len,
            theta=config.rope_theta,
        )
        self.layers = nn.ModuleList([
            LlamaTransformerBlock(config, i, self.pe)
            for i in range(config.n_layers)
        ])
        self.norm = RMSNorm(normalized_shape=config.d_model, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(self, tokens, kv_cache=None, position_ids=None):
        x = self.embed(tokens)
        for i, layer in enumerate(self.layers):
            layer_kv = kv_cache[i] if kv_cache is not None else None
            x = layer(x, kv_cache=layer_kv, position_ids=position_ids)
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits
# ENDANCHOR: LlamaModel
# ─── ENDSECTION: Llama Model ────────────────────────────
