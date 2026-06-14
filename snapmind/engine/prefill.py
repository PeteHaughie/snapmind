# ─── SECTION: Prefill ───────────────────────────────────
import time

import torch


# ANCHOR: prefill
def prefill(model, tokens, kv_cache):
    model.eval()
    with torch.no_grad():
        t0 = time.perf_counter()
        logits = model(tokens, kv_cache=kv_cache)
        t1 = time.perf_counter()
    last_logits = logits[:, -1, :]
    ttft = t1 - t0
    return last_logits, ttft
# ENDANCHOR: prefill
# ─── ENDSECTION: Prefill ────────────────────────────────
