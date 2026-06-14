# ─── SECTION: Temperature Sampler ───────────────────────
from snapmind.core.registry import SAMPLER
from snapmind.sampling.base import SamplerABC
from snapmind.sampling.greedy import GreedySampler


# ANCHOR: TemperatureSampler
@SAMPLER.register("temperature")
class TemperatureSampler(SamplerABC):
    def __init__(self, wrapped: str = "greedy"):
        from snapmind.core.registry import SAMPLER as _S
        self._wrapped = _S.create(wrapped)

    def sample(self, logits, temperature=1.0, **kwargs):
        if temperature == 0.0 or temperature < 1e-8:
            return logits.argmax(dim=-1)
        scaled = logits / temperature
        return self._wrapped.sample(scaled, temperature=1.0, **kwargs)
# ENDANCHOR: TemperatureSampler
# ─── ENDSECTION: Temperature Sampler ─────────────────────
