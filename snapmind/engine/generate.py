# ─── SECTION: Generate Engine ───────────────────────────
from collections.abc import AsyncIterator
from typing import Any

import torch
import torch.nn as nn

from snapmind.core.config import EngineConfig
from snapmind.engine.decode import decode_step
from snapmind.engine.prefill import prefill
from snapmind.kv_cache.base import KVCacheABC
from snapmind.sampling.base import SamplerABC
from snapmind.tokenizer.base import TokenizerABC


# ANCHOR: GenerateEngine
class GenerateEngine:
    def __init__(
        self,
        model: nn.Module,
        tokenizer: TokenizerABC,
        sampler: SamplerABC,
        eos_token_id: int | None = None,
        engine_config: EngineConfig | None = None,
        kvcache: KVCacheABC | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.sampler = sampler
        self._eos_token_id = eos_token_id if eos_token_id is not None else getattr(tokenizer, "eos_token_id", 50256)
        self._engine_config = engine_config or EngineConfig()
        if kvcache is not None:
            self._kvcache = kvcache
        else:
            from snapmind.kv_cache.naive import NaiveKVCache

            cfg: Any = self.model.config
            self._kvcache = NaiveKVCache(
                max_seq_len=cfg.max_seq_len,
                n_layers=cfg.n_layers,
                n_heads=cfg.n_heads,
                head_dim=cfg.d_model // cfg.n_heads,
            )

    def _make_kv_cache(self) -> dict:
        self._kvcache.reset()
        return self._kvcache.layer_dicts()

    async def generate(
        self,
        prompt: str,
        max_tokens: int | None = None,
        temperature: float = 1.0,
        **sampler_kwargs,
    ) -> AsyncIterator[str]:
        if max_tokens is None:
            max_tokens = self._engine_config.max_tokens
        kv_cache = self._make_kv_cache()
        input_ids = self.tokenizer.encode(prompt)
        device = next(self.model.parameters()).device
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

        last_logits, _ = prefill(self.model, input_tensor, kv_cache)
        sampled = self.sampler.sample(last_logits[0, :], temperature=temperature, **sampler_kwargs)
        next_token_id = int(sampled.item())

        step = len(input_ids)
        for _ in range(max_tokens):
            if next_token_id == self._eos_token_id:
                break
            token_str = self.tokenizer.decode([next_token_id])
            yield token_str
            next_token_id = decode_step(
                self.model,
                next_token_id,
                kv_cache,
                self.sampler,
                temperature=temperature,
                position_ids=torch.tensor([step], device=device),
                **sampler_kwargs,
            )
            step += 1


# ENDANCHOR: GenerateEngine
# ─── ENDSECTION: Generate Engine ────────────────────────
