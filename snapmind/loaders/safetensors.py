# ─── SECTION: Safetensors Loader ────────────────────────
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from snapmind.core.registry import LOADER
from snapmind.loaders.base import WeightLoaderABC


REPO_ID = "openai-community/gpt2"
SAFETENSORS_FILE = "model.safetensors"

HF_TO_SNAPMIND = {
    "wte.weight": "embed.weight",
    "ln_f.weight": "ln_f.weight",
    "ln_f.bias": "ln_f.bias",
}


def _block_map(layer_idx: int) -> dict:
    return {
        f"h.{layer_idx}.ln_1.weight": f"layers.{layer_idx}.ln1.weight",
        f"h.{layer_idx}.ln_1.bias": f"layers.{layer_idx}.ln1.bias",
        f"h.{layer_idx}.ln_2.weight": f"layers.{layer_idx}.ln2.weight",
        f"h.{layer_idx}.ln_2.bias": f"layers.{layer_idx}.ln2.bias",
        f"h.{layer_idx}.attn.c_proj.weight": f"layers.{layer_idx}.self_attn.out_proj.weight",
        f"h.{layer_idx}.attn.c_proj.bias": f"layers.{layer_idx}.self_attn.out_proj.bias",
        f"h.{layer_idx}.mlp.c_fc.weight": f"layers.{layer_idx}.feed_forward.gate_proj.weight",
        f"h.{layer_idx}.mlp.c_fc.bias": f"layers.{layer_idx}.feed_forward.gate_proj.bias",
        f"h.{layer_idx}.mlp.c_proj.weight": f"layers.{layer_idx}.feed_forward.down_proj.weight",
        f"h.{layer_idx}.mlp.c_proj.bias": f"layers.{layer_idx}.feed_forward.down_proj.bias",
    }


_CONV1D_WEIGHT_KEYS = {
    "attn.c_proj.weight",
    "mlp.c_fc.weight",
    "mlp.c_proj.weight",
}


def _load_and_map_state_dict(model, config, state_dict):
    new_state = {}
    for hf_key, local_key in HF_TO_SNAPMIND.items():
        if hf_key in state_dict:
            new_state[local_key] = state_dict[hf_key]
    for i in range(config.n_layers):
        for hf_key, local_key in _block_map(i).items():
            if hf_key not in state_dict:
                continue
            tensor = state_dict[hf_key]
            weight_key = hf_key.split(f"h.{i}.")[1]
            if weight_key in _CONV1D_WEIGHT_KEYS:
                tensor = tensor.t()
            new_state[local_key] = tensor
    if "wpe.weight" in state_dict and hasattr(model, "pe") and hasattr(model.pe, "pe"):
        new_state["pe.pe.weight"] = state_dict["wpe.weight"]
    for i in range(config.n_layers):
        hf_w = f"h.{i}.attn.c_attn.weight"
        hf_b = f"h.{i}.attn.c_attn.bias"
        if hf_w not in state_dict:
            continue
        combined_w = state_dict[hf_w]
        combined_b = state_dict.get(hf_b)
        d_model = config.d_model
        w_t = combined_w.t()
        new_state[f"layers.{i}.self_attn.q_proj.weight"] = w_t[:d_model]
        new_state[f"layers.{i}.self_attn.k_proj.weight"] = w_t[d_model:2*d_model]
        new_state[f"layers.{i}.self_attn.v_proj.weight"] = w_t[2*d_model:]
        if combined_b is not None:
            new_state[f"layers.{i}.self_attn.q_proj.bias"] = combined_b[:d_model]
            new_state[f"layers.{i}.self_attn.k_proj.bias"] = combined_b[d_model:2*d_model]
            new_state[f"layers.{i}.self_attn.v_proj.bias"] = combined_b[2*d_model:]
    return new_state


# ANCHOR: SafetensorsLoader
@LOADER.register("safetensors")
class SafetensorsLoader(WeightLoaderABC):
    def load(self, path, model, config):
        if path is None:
            path = hf_hub_download(repo_id=REPO_ID, filename=SAFETENSORS_FILE)
        state_dict = load_file(path, device="cpu")
        mapped = _load_and_map_state_dict(model, config, state_dict)
        missing, unexpected = model.load_state_dict(mapped, strict=False)
        return {"missing": missing, "unexpected": unexpected, "path": path}
# ENDANCHOR: SafetensorsLoader

# ANCHOR: SafetensorsDownloadLoader
@LOADER.register("safetensors_download")
class SafetensorsDownloadLoader(WeightLoaderABC):
    def load(self, path, model, config):
        local_path = hf_hub_download(repo_id=REPO_ID, filename=SAFETENSORS_FILE)
        loader = SafetensorsLoader()
        return loader.load(local_path, model, config)
# ENDANCHOR: SafetensorsDownloadLoader
# ─── ENDSECTION: Safetensors Loader ─────────────────────
