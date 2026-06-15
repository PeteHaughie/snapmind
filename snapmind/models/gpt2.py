# ─── SECTION: GPT-2 Model ───────────────────────────────
import torch
import torch.nn as nn

from snapmind.core.architecture import ARCHITECTURE, SupportedArchitecture
from snapmind.core.config import ModelConfig
from snapmind.layers.positional.learned import LearnedPositionalEncoding
from snapmind.layers.transformer_block import TransformerBlock
from snapmind.models.base import BaseModelABC


# ANCHOR: GPT2Model
class GPT2Model(BaseModelABC):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.pe = LearnedPositionalEncoding(
            d_model=config.d_model,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout,
        )
        self.layers = nn.ModuleList([TransformerBlock(config, layer_idx=i) for i in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.d_model, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight

    def forward(
        self, tokens: torch.Tensor, kv_cache: dict | None = None, position_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = self.embed(tokens)
        x = self.pe(x, position_ids=position_ids)
        for i, layer in enumerate(self.layers):
            layer_kv = kv_cache[i] if kv_cache is not None else None
            x = layer(x, kv_cache=layer_kv, position_ids=position_ids)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits


# ENDANCHOR: GPT2Model

ARCHITECTURE.register(
    "gpt2",
    SupportedArchitecture(
        name="gpt2",
        model_cls=GPT2Model,
        hf_repo="openai-community/gpt2",
        default_config=dict(d_model=768, n_heads=12, n_layers=12, vocab_size=50257, max_seq_len=1024, model_type="gpt2"),
    ),
)
# ─── ENDSECTION: GPT-2 Model ────────────────────────────
