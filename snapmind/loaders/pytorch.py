# ─── SECTION: PyTorch Loader ─────────────────────────
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

from snapmind.core.config import ModelConfig
from snapmind.core.registry import LOADER
from snapmind.loaders.base import WeightLoaderABC
from snapmind.loaders.safetensors import _HF_MODEL_MAP


# ANCHOR: PyTorchLoader
@LOADER.register("pytorch")
class PyTorchLoader(WeightLoaderABC):
    def load(self, path: str | None, model: nn.Module, config: ModelConfig) -> dict:
        info = _HF_MODEL_MAP.get(config.model_type)
        if info is None:
            raise ValueError(f"No HF model mapping for '{config.model_type}'")
        repo_id, _filename, remap_fn = info
        bin_filename = _filename.replace(".safetensors", ".bin")
        if path is None:
            path = hf_hub_download(repo_id=repo_id, filename=bin_filename)
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        mapped = remap_fn(state_dict, model, config)
        missing, unexpected = model.load_state_dict(mapped, strict=False)
        return {"missing": missing, "unexpected": unexpected, "path": path}


# ENDANCHOR: PyTorchLoader
# ─── ENDSECTION: PyTorch Loader ──────────────────────
