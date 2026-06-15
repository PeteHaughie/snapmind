# ─── SECTION: Sampler ABC ───────────────────────────────
import abc

import torch


# ANCHOR: SamplerABC
class SamplerABC(abc.ABC):
    """Base class for sampling strategies (greedy, temperature, top-k, top-p, mirostat, …).

    Subclasses implement :meth:`sample` to select a token index from the logit distribution.
    """

    @abc.abstractmethod
    def sample(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0,
        **kwargs,
    ) -> torch.Tensor:
        """Select a token index given raw logits.

        Args:
            logits: Unnormalized scores ``(batch, vocab_size)`` or ``(vocab_size,)``.
            temperature: Sampling temperature. Values near 0 encourage greediness.
            **kwargs: Strategy-specific parameters (``top_k``, ``top_p``, etc.).

        Returns:
            Selected token index tensor (0-d for single, 1-d for batched).
        """


# ENDANCHOR: SamplerABC
# ─── ENDSECTION: Sampler ABC ────────────────────────────
