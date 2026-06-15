# ─── SECTION: Mistral Model ──────────────────────────────
import torch
import torch.nn as nn

from snapmind.core.architecture import ARCHITECTURE, SupportedArchitecture
from snapmind.core.config import ModelConfig
from snapmind.core.registry import MODEL
from snapmind.layers.attention.gqa import GroupedQueryAttention
from snapmind.layers.gated_feed_forward import GatedFeedForward
from snapmind.layers.normalization.rms_norm import RMSNorm
from snapmind.layers.positional.rope import RotaryPositionalEncoding
from snapmind.models.base import BaseModelABC


def create_sliding_window_mask(seq_len: int, device: torch.device, window_size: int) -> torch.Tensor:
    if window_size <= 0:
        return torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=device), diagonal=1)
    mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
    for i in range(seq_len):
        start = max(0, i - window_size + 1)
        mask[i, start : i + 1] = 0.0
    return mask


# ANCHOR: MistralTransformerBlock
class MistralTransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int, pe: RotaryPositionalEncoding, window_size: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.window_size = window_size
        d_ff: int = config.d_ff if config.d_ff is not None else config.d_model * 4
        self.self_attn = GroupedQueryAttention(
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_kv_heads=config.n_kv_heads,
            dropout=config.dropout,
            bias=False,
            pe=pe,
            head_dim=config.head_dim,
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
        if is_prefill:
            mask = create_sliding_window_mask(x.shape[1], x.device, self.window_size)
        else:
            mask = None
        attn_out, _ = self.self_attn(self.input_layernorm(x), kv_cache=kv_cache, mask=mask, position_ids=position_ids)
        if kv_cache is not None and kv_cache.get("k") is not None and self.window_size > 0:
            kv_cache["k"] = kv_cache["k"][..., -self.window_size :, :]
            kv_cache["v"] = kv_cache["v"][..., -self.window_size :, :]
        x = x + attn_out
        ffn_out = self.mlp(self.post_attention_layernorm(x))
        x = x + ffn_out
        return x


# ENDANCHOR: MistralTransformerBlock


# ANCHOR: MistralModel
@MODEL.register("mistral")
class MistralModel(BaseModelABC):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.config = config
        self.window_size = getattr(config, "window_size", 0)
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        head_dim = config.head_dim if config.head_dim is not None else config.d_model // config.n_heads
        self.pe = RotaryPositionalEncoding(
            dim=head_dim,
            max_seq_len=config.max_seq_len,
            theta=config.rope_theta,
        )
        self.layers = nn.ModuleList(
            [MistralTransformerBlock(config, i, self.pe, self.window_size) for i in range(config.n_layers)]
        )
        self.norm = RMSNorm(normalized_shape=config.d_model, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if getattr(config, "tie_word_embeddings", False):
            self.lm_head.weight = self.embed.weight

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


# ENDANCHOR: MistralModel

ARCHITECTURE.register(
    "mistral",
    SupportedArchitecture(
        name="mistral",
        model_cls=MistralModel,
        hf_repo="mistralai/Mistral-7B-v0.1",
        default_config=dict(
            d_model=4096, n_heads=32, n_kv_heads=8, n_layers=32, vocab_size=32000,
            max_seq_len=8192, d_ff=14336, norm_eps=1e-05, rope_theta=10000.0, window_size=4096,
        ),
    ),
)
ARCHITECTURE.register(
    "ministral-3-3b",
    SupportedArchitecture(
        name="ministral-3-3b",
        model_cls=MistralModel,
        hf_repo="mistralai/Ministral-3-3B-Base-2512",
        hf_filename="consolidated.safetensors",
        default_config=dict(
            d_model=3072, n_heads=32, n_kv_heads=8, n_layers=26, vocab_size=131072,
            max_seq_len=262144, d_ff=9216, norm_eps=1e-05, rope_theta=1000000.0,
            window_size=0, head_dim=128, tie_word_embeddings=True,
        ),
    ),
)
# ─── ENDSECTION: Mistral Model ───────────────────────────
