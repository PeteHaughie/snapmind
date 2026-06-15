# ─── SECTION: Base Model ABC ─────────────────────────────
import abc

import torch
import torch.nn as nn

from snapmind.core.config import ModelConfig


# ANCHOR: BaseModelABC
class BaseModelABC(nn.Module, abc.ABC):
    """Base class for transformer models (GPT-2, Llama, Mistral, …).

    Stores ``config`` on the instance. Subclasses build their own layer stack
    and implement :meth:`forward`.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

    @abc.abstractmethod
    def forward(
        self, tokens: torch.Tensor, kv_cache: dict | None = None, position_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Run the full model on *tokens*, returning logits ``(batch, seq, vocab_size)``."""


# ENDANCHOR: BaseModelABC
# ─── ENDSECTION: Base Model ABC ──────────────────────────
