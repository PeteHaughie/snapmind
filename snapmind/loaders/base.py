# ─── SECTION: Weight Loader ABC ─────────────────────────
import abc

import torch.nn as nn

from snapmind.core.config import ModelConfig


# ANCHOR: WeightLoaderABC
class WeightLoaderABC(abc.ABC):
    """Base class for weight loaders (safetensors, PyTorch, …).

    Subclasses implement :meth:`load` to populate a model from a checkpoint path.
    """

    @abc.abstractmethod
    def load(self, path: str | None, model: nn.Module, config: ModelConfig) -> dict:
        """Load weights from *path* into *model*, returning a state dict.

        Args:
            path: Local file path, or ``None`` to download from HuggingFace.
            model: The model to populate.
            config: Model configuration (used for key remapping).

        Returns:
            The loaded state dict.
        """


# ENDANCHOR: WeightLoaderABC
# ─── ENDSECTION: Weight Loader ABC ──────────────────────
