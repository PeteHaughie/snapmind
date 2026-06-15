# ─── SECTION: Mirostat Sampler ──────────────────────────
import math

import torch
import torch.nn.functional as F

from snapmind.core.registry import SAMPLER
from snapmind.sampling.base import SamplerABC


# ANCHOR: MirostatSampler
@SAMPLER.register("mirostat")
class MirostatSampler(SamplerABC):
    def __init__(self, tau: float = 5.0, learning_rate: float = 0.1, max_surprise: float = 2.0):
        self.tau = tau
        self.rate = learning_rate
        self.max_surprise = max_surprise

    def reset(self) -> None:
        self.max_surprise = 2.0

    def sample(
        self, logits: torch.Tensor, temperature: float = 1.0, **kwargs
    ) -> int | torch.Tensor:
        if temperature == 0.0 or temperature < 1e-8:
            return logits.argmax(dim=-1)

        probs = F.softmax(logits / temperature, dim=-1)
        sorted_probs, sorted_indices = probs.sort(descending=True, dim=-1)
        k = self._estimate_k(sorted_probs)

        top_probs = sorted_probs[..., :k]
        top_indices = sorted_indices[..., :k]
        top_probs = top_probs / top_probs.sum(dim=-1, keepdim=True).clamp(min=1e-10)

        flat = top_probs.reshape(-1, top_probs.size(-1))
        idx = flat.multinomial(1, replacement=True)
        idx = idx.reshape(*top_probs.shape[:-1], 1)
        token = torch.gather(top_indices, -1, idx).squeeze(-1)

        self._update_surprise(probs, token)
        return token

    def _estimate_k(self, sorted_probs: torch.Tensor) -> int:
        vocab_size = sorted_probs.size(-1)
        cutoff = 1.0 / (math.e**self.max_surprise)
        k = (sorted_probs > cutoff).sum(dim=-1).max().item()
        return max(1, min(int(k), vocab_size - 1))

    def _update_surprise(self, probs: torch.Tensor, token: torch.Tensor) -> None:
        p = probs.gather(-1, token.unsqueeze(-1)).squeeze(-1)
        observed = -torch.log2(p.clamp(min=1e-10))
        self.max_surprise += self.rate * (self.tau - observed.mean().item())
        self.max_surprise = max(0.1, min(self.max_surprise, 10.0))


# ENDANCHOR: MirostatSampler
# ─── ENDSECTION: Mirostat Sampler ──────────────────────────
