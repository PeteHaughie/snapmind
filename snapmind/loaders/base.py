# ─── SECTION: Weight Loader ABC ─────────────────────────
import abc

import torch.nn as nn

from snapmind.core.config import ModelConfig


# ANCHOR: WeightLoaderABC
class WeightLoaderABC(abc.ABC):
    @abc.abstractmethod
    def load(self, path: str | None, model: nn.Module, config: ModelConfig) -> dict: ...


# ENDANCHOR: WeightLoaderABC
# ─── ENDSECTION: Weight Loader ABC ──────────────────────
