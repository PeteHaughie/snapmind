# ─── SECTION: Device resolution ─────────────────────────
import torch


def resolve_device(device: str = "auto") -> torch.device:
    """Return a ``torch.device`` preferring MPS → CUDA → CPU.

    Args:
        device: ``"auto"`` (default) or an explicit device string (``"mps"``, ``"cuda"``, ``"cpu"``).
    """
    if device != "auto":
        return torch.device(device)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_model_device(model: torch.nn.Module) -> torch.device:
    """Return the device of the first parameter in *model*."""
    return next(model.parameters()).device


# ─── ENDSECTION: Device resolution ──────────────────────
