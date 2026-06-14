# ─── SECTION: Base Model ABC ─────────────────────────────
import abc
import torch.nn as nn
from snapmind.core.config import ModelConfig


# ANCHOR: BaseModelABC
class BaseModelABC(nn.Module, abc.ABC):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

    @abc.abstractmethod
    def forward(self, tokens, kv_cache=None, position_ids=None):
        ...
# ENDANCHOR: BaseModelABC
# ─── ENDSECTION: Base Model ABC ──────────────────────────
