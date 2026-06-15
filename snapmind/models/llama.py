# ─── SECTION: Llama Model ───────────────────────────────
import torch
import torch.nn as nn

from snapmind.core.architecture import ARCHITECTURE, SupportedArchitecture
from snapmind.core.config import ModelConfig
from snapmind.layers.attention.gqa import GroupedQueryAttention
from snapmind.layers.gated_feed_forward import GatedFeedForward
from snapmind.layers.normalization.rms_norm import RMSNorm
from snapmind.layers.positional.rope import RotaryPositionalEncoding
from snapmind.models.base import BaseModelABC


# ANCHOR: LlamaTransformerBlock
class LlamaTransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int, pe: RotaryPositionalEncoding):
        super().__init__()
        self.layer_idx = layer_idx
        d_ff: int = config.d_ff if config.d_ff is not None else config.d_model * 4
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
            d_ff=d_ff,
            dropout=config.dropout,
        )

    def forward(
        self, x: torch.Tensor, kv_cache: dict | None = None, position_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
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
        self.layers = nn.ModuleList([LlamaTransformerBlock(config, i, self.pe) for i in range(config.n_layers)])
        self.norm = RMSNorm(normalized_shape=config.d_model, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(
        self, tokens: torch.Tensor, kv_cache: dict | None = None, position_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = self.embed(tokens)
        for i, layer in enumerate(self.layers):
            layer_kv = kv_cache[i] if kv_cache is not None else None
            x = layer(x, kv_cache=layer_kv, position_ids=position_ids)
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits


# ENDANCHOR: LlamaModel

ARCHITECTURE.register(
    "tinyllama",
    SupportedArchitecture(
        name="tinyllama",
        model_cls=LlamaModel,
        hf_repo="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        default_config=dict(
            d_model=2048, n_heads=32, n_kv_heads=4, n_layers=22, vocab_size=32000,
            max_seq_len=2048, d_ff=5632, norm_eps=1e-05, rope_theta=10000.0,
        ),
    ),
)
ARCHITECTURE.register(
    "llama",
    SupportedArchitecture(
        name="llama",
        model_cls=LlamaModel,
        hf_repo=None,
        default_config=dict(
            d_model=4096, n_heads=32, n_kv_heads=8, n_layers=32, vocab_size=32000, max_seq_len=8192,
        ),
    ),
)
# ─── ENDSECTION: Llama Model ────────────────────────────
