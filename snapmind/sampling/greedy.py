# ─── SECTION: Greedy Sampler ────────────────────────────
from snapmind.core.registry import SAMPLER
from snapmind.sampling.base import SamplerABC


# ANCHOR: GreedySampler
@SAMPLER.register("greedy")
class GreedySampler(SamplerABC):
    def sample(self, logits, temperature=1.0):
        return logits.argmax(dim=-1)
# ENDANCHOR: GreedySampler
# ─── ENDSECTION: Greedy Sampler ─────────────────────────
