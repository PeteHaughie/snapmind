# ─── SECTION: Activation ABC ─────────────────────────────
import abc
import torch.nn as nn


# ANCHOR: ActivationABC
class ActivationABC(nn.Module, abc.ABC):
    @abc.abstractmethod
    def forward(self, x):
        ...
# ENDANCHOR: ActivationABC
# ─── ENDSECTION: Activation ABC ──────────────────────────
