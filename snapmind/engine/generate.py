# ─── SECTION: Generate Engine ───────────────────────────
from collections.abc import AsyncIterator
from typing import Any

import torch
import torch.nn as nn

from snapmind.engine.decode import decode_step
from snapmind.engine.prefill import prefill
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
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.sampler = sampler
        self._eos_token_id = eos_token_id if eos_token_id is not None else getattr(tokenizer, "eos_token_id", 50256)

    def _make_kv_cache(self) -> dict:
        cfg: Any = self.model.config
        return {i: {"k": None, "v": None} for i in range(cfg.n_layers)}

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 1.0,
        **sampler_kwargs,
    ) -> AsyncIterator[str]:
        kv_cache = self._make_kv_cache()
        input_ids = self.tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_ids], dtype=torch.long)

        last_logits, _ = prefill(self.model, input_tensor, kv_cache)
        sampled = self.sampler.sample(last_logits[0, :], temperature=temperature, **sampler_kwargs)
        if isinstance(sampled, torch.Tensor):
            next_token_id = int(sampled.item())
        else:
            next_token_id = sampled

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
                position_ids=torch.tensor([step]),
                **sampler_kwargs,
            )
            step += 1


# ENDANCHOR: GenerateEngine
# ─── ENDSECTION: Generate Engine ────────────────────────
