# ─── SECTION: PyTorch Loader ─────────────────────────
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

from snapmind.core.architecture import ARCHITECTURE
from snapmind.core.config import ModelConfig
from snapmind.core.registry import LOADER
from snapmind.loaders.base import WeightLoaderABC
from snapmind.loaders.safetensors import _REMAP_FN


# ANCHOR: PyTorchLoader
@LOADER.register("pytorch")
class PyTorchLoader(WeightLoaderABC):
    def load(self, path: str | None, model: nn.Module, config: ModelConfig) -> dict:
        arch = ARCHITECTURE.get(config.model_type)
        if arch.hf_repo is None:
            return {"missing": [], "unexpected": [], "path": None}

        remap_fn = _REMAP_FN.get(config.model_type)
        if remap_fn is None:
            raise ValueError(f"No weight remap function for '{config.model_type}'")

        repo_id: str = arch.hf_repo
        _filename: str = arch.hf_filename
        bin_filename = _filename.replace(".safetensors", ".bin")
        if path is None:
            path = hf_hub_download(repo_id=repo_id, filename=bin_filename)
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        mapped = remap_fn(state_dict, model, config)
        missing, unexpected = model.load_state_dict(mapped, strict=False)
        return {"missing": missing, "unexpected": unexpected, "path": path}


# ENDANCHOR: PyTorchLoader
# ─── ENDSECTION: PyTorch Loader ──────────────────────
