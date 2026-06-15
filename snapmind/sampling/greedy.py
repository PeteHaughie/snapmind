# ─── SECTION: Greedy Sampler ────────────────────────────
import torch

from snapmind.core.registry import SAMPLER
from snapmind.sampling.base import SamplerABC


# ANCHOR: GreedySampler
@SAMPLER.register("greedy")
class GreedySampler(SamplerABC):
    def sample(self, logits: torch.Tensor, temperature: float = 1.0, **kwargs) -> torch.Tensor:
        return logits.argmax(dim=-1)


# ENDANCHOR: GreedySampler
# ─── ENDSECTION: Greedy Sampler ─────────────────────────
