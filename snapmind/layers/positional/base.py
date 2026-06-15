# ─── SECTION: Positional Encoding ABC ────────────────────
import abc

import torch
import torch.nn as nn


# ANCHOR: PositionalEncodingABC
class PositionalEncodingABC(nn.Module, abc.ABC):
    """Base class for positional encodings (learned, RoPE, none, …).

    Subclasses declare an ``injection_point`` (``"residual"`` or ``"attention"``) and
    implement :meth:`forward`. RoPE-style encodings additionally override :meth:`apply_to_qk`.
    """

    @property
    @abc.abstractmethod
    def injection_point(self) -> str:
        """Where the encoding is applied: ``"residual"`` (added to the residual stream)
        or ``"attention"`` (applied in Q/K space)."""

    @abc.abstractmethod
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Apply the positional encoding to the input tensor."""

    def apply_to_qk(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary-style position encoding to Q and K tensors in-place.

        Default no-op — override in subclasses that mutate Q/K inside attention.
        """
        return q, k


# ENDANCHOR: PositionalEncodingABC
# ─── ENDSECTION: Positional Encoding ABC ─────────────────
