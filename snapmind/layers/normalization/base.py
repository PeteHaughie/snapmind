# ─── SECTION: Norm ABC ──────────────────────────────────
import abc
import torch.nn as nn


# ANCHOR: NormABC
class NormABC(nn.Module, abc.ABC):
    @abc.abstractmethod
    def forward(self, x):
        ...
# ENDANCHOR: NormABC
# ─── ENDSECTION: Norm ABC ────────────────────────────────
