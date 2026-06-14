# ─── SECTION: Device resolution ─────────────────────────
import torch


def resolve_device(device: str = "auto") -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_model_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


# ─── ENDSECTION: Device resolution ──────────────────────
