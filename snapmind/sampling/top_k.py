# ─── SECTION: Top-K Sampler ─────────────────────────────
import torch
import torch.nn.functional as F

from snapmind.core.registry import SAMPLER
from snapmind.sampling.base import SamplerABC


# ANCHOR: TopKSampler
@SAMPLER.register("top_k")
class TopKSampler(SamplerABC):
    def sample(
        self, logits: torch.Tensor, temperature: float = 1.0, top_k: int = 50, **kwargs
    ) -> int | torch.Tensor:
        if temperature == 0.0 or temperature < 1e-8:
            return logits.argmax(dim=-1)

        logits = logits / temperature
        if top_k > 0:
            k = min(top_k, logits.size(-1))
            threshold = logits.topk(k, dim=-1).values[..., -1, None]
            logits = torch.where(logits >= threshold, logits, float("-inf"))

        probs = F.softmax(logits, dim=-1)
        flat = probs.reshape(-1, probs.size(-1))
        idx = flat.multinomial(1, replacement=True)
        idx = idx.reshape(*probs.shape[:-1], 1)
        return idx.squeeze(-1)


# ENDANCHOR: TopKSampler
# ─── ENDSECTION: Top-K Sampler ──────────────────────────
