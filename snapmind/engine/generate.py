# ─── SECTION: Generate Engine ───────────────────────────
from collections.abc import AsyncIterator

import torch

from snapmind.engine.prefill import prefill
from snapmind.engine.decode import decode_step


# ANCHOR: GenerateEngine
class GenerateEngine:
    def __init__(self, model, tokenizer, sampler, eos_token_id=None):
        self.model = model
        self.tokenizer = tokenizer
        self.sampler = sampler
        self._eos_token_id = eos_token_id if eos_token_id is not None else getattr(tokenizer, "eos_token_id", 50256)

    def _make_kv_cache(self):
        cfg = self.model.config
        head_dim = cfg.d_model // cfg.n_heads
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
        next_token_id = self.sampler.sample(
            last_logits[0, :], temperature=temperature, **sampler_kwargs
        )
        if isinstance(next_token_id, torch.Tensor):
            next_token_id = next_token_id.item()

        step = len(input_ids)
        for _ in range(max_tokens):
            if next_token_id == self._eos_token_id:
                break
            token_str = self.tokenizer.decode([next_token_id])
            yield token_str
            next_token_id = decode_step(
                self.model, next_token_id, kv_cache,
                self.sampler, temperature=temperature,
                position_ids=torch.tensor([step]),
                **sampler_kwargs,
            )
            step += 1
# ENDANCHOR: GenerateEngine
# ─── ENDSECTION: Generate Engine ────────────────────────
