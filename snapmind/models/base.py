# ─── SECTION: Base Model ABC ─────────────────────────────
import abc

import torch
import torch.nn as nn

from snapmind.core.config import ModelConfig


# ANCHOR: BaseModelABC
class BaseModelABC(nn.Module, abc.ABC):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

    @abc.abstractmethod
    def forward(
        self, tokens: torch.Tensor, kv_cache: dict | None = None, position_ids: torch.Tensor | None = None
    ) -> torch.Tensor: ...


# ENDANCHOR: BaseModelABC
# ─── ENDSECTION: Base Model ABC ──────────────────────────
