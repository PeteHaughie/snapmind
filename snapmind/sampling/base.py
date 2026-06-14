# ─── SECTION: Sampler ABC ───────────────────────────────
import abc


# ANCHOR: SamplerABC
class SamplerABC(abc.ABC):
    @abc.abstractmethod
    def sample(self, logits, **kwargs):
        ...
# ENDANCHOR: SamplerABC
# ─── ENDSECTION: Sampler ABC ────────────────────────────
