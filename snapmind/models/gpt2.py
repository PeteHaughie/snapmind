# ─── SECTION: GPT-2 Model ───────────────────────────────
import torch.nn as nn
from snapmind.core.config import ModelConfig
from snapmind.core.registry import PE, ATTENTION
from snapmind.models.base import BaseModelABC
from snapmind.layers.positional.learned import LearnedPositionalEncoding
from snapmind.layers.transformer_block import TransformerBlock


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
        self.layers = nn.ModuleList([
            TransformerBlock(config, layer_idx=i)
            for i in range(config.n_layers)
        ])
        self.ln_f = nn.LayerNorm(config.d_model, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight

    def forward(self, tokens, kv_cache=None, position_ids=None):
        x = self.embed(tokens)
        x = self.pe(x, position_ids=position_ids)
        for i, layer in enumerate(self.layers):
            layer_kv = kv_cache[i] if kv_cache is not None else None
            x = layer(x, kv_cache=layer_kv, position_ids=position_ids)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits
# ENDANCHOR: GPT2Model
# ─── ENDSECTION: GPT-2 Model ────────────────────────────
