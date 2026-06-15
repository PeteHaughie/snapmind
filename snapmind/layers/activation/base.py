# ─── SECTION: Activation ABC ─────────────────────────────
import abc

import torch
import torch.nn as nn


# ANCHOR: ActivationABC
class ActivationABC(nn.Module, abc.ABC):
    """Base class for activation functions (GELU, SiLU, …).

    Subclasses must implement :meth:`forward` as a pointwise function.
    """

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the activation function element-wise."""


# ENDANCHOR: ActivationABC
# ─── ENDSECTION: Activation ABC ──────────────────────────
