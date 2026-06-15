# ─── SECTION: Top-P Sampler ─────────────────────────────
import torch
import torch.nn.functional as F

from snapmind.core.registry import SAMPLER
from snapmind.sampling.base import SamplerABC


# ANCHOR: TopPSampler
@SAMPLER.register("top_p")
class TopPSampler(SamplerABC):
    def sample(
        self, logits: torch.Tensor, temperature: float = 1.0, top_p: float = 0.9, top_k: int = 0, **kwargs
    ) -> torch.Tensor:
        if temperature == 0.0 or temperature < 1e-8:
            return logits.argmax(dim=-1)

        logits = logits / temperature
        probs = F.softmax(logits, dim=-1)
        sorted_probs, sorted_indices = probs.sort(descending=True, dim=-1)
        cumsum = sorted_probs.cumsum(dim=-1)

        mask = cumsum - sorted_probs > top_p
        sorted_probs[mask] = 0.0
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp(min=1e-10)

        batch_shape = sorted_probs.shape[:-1]
        flat = sorted_probs.reshape(-1, sorted_probs.size(-1))
        idx = flat.multinomial(1, replacement=True)
        idx = idx.reshape(*batch_shape, 1)
        gathered = sorted_indices.gather(-1, idx)
        return gathered.squeeze(-1)


# ENDANCHOR: TopPSampler
# ─── ENDSECTION: Top-P Sampler ──────────────────────────
