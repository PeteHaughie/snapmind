# ─── SECTION: Decode ────────────────────────────────────
import torch
import torch.nn as nn

from snapmind.sampling.base import SamplerABC


# ANCHOR: decode_step
def decode_step(
    model: nn.Module,
    last_token_id: int,
    kv_cache: dict,
    sampler: SamplerABC,
    temperature: float = 1.0,
    position_ids: torch.Tensor | None = None,
    **sampler_kwargs,
) -> int:
    model.eval()
    token_tensor = torch.tensor([[last_token_id]], dtype=torch.long)
    with torch.no_grad():
        logits = model(token_tensor, kv_cache=kv_cache, position_ids=position_ids)
    next_logits = logits[0, -1, :]
    next_token_id = sampler.sample(next_logits, temperature=temperature, **sampler_kwargs)
    if isinstance(next_token_id, torch.Tensor):
        next_token_id = int(next_token_id.item())
    return next_token_id


# ENDANCHOR: decode_step
# ─── ENDSECTION: Decode ─────────────────────────────────
