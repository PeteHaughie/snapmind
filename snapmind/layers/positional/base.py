# ─── SECTION: Positional Encoding ABC ────────────────────
import abc
import torch
import torch.nn as nn


# ANCHOR: PositionalEncodingABC
class PositionalEncodingABC(nn.Module, abc.ABC):
    @property
    @abc.abstractmethod
    def injection_point(self):
        ...

    @abc.abstractmethod
    def forward(self, x, position_ids=None):
        ...

    def apply_to_qk(self, q: torch.Tensor, k: torch.Tensor, position_ids: torch.Tensor | None = None):
        return q, k
# ENDANCHOR: PositionalEncodingABC
# ─── ENDSECTION: Positional Encoding ABC ─────────────────
