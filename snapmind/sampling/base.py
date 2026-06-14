# ─── SECTION: Sampler ABC ───────────────────────────────
import abc

import torch


# ANCHOR: SamplerABC
class SamplerABC(abc.ABC):
    @abc.abstractmethod
    def sample(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0,
        **kwargs,
    ) -> int | torch.Tensor: ...


# ENDANCHOR: SamplerABC
# ─── ENDSECTION: Sampler ABC ────────────────────────────
