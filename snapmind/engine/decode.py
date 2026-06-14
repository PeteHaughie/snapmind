# ─── SECTION: Decode ────────────────────────────────────
import torch


# ANCHOR: decode_step
def decode_step(model, last_token_id, kv_cache, sampler, temperature=1.0, **sampler_kwargs):
    model.eval()
    token_tensor = torch.tensor([[last_token_id]], dtype=torch.long)
    with torch.no_grad():
        logits = model(token_tensor, kv_cache=kv_cache)
    next_logits = logits[0, -1, :]
    next_token_id = sampler.sample(next_logits, temperature=temperature, **sampler_kwargs)
    if isinstance(next_token_id, torch.Tensor):
        next_token_id = next_token_id.item()
    return next_token_id
# ENDANCHOR: decode_step
# ─── ENDSECTION: Decode ─────────────────────────────────
