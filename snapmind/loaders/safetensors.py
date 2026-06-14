# ─── SECTION: Safetensors Loader ────────────────────────
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from snapmind.core.registry import LOADER
from snapmind.loaders.base import WeightLoaderABC


_HF_MODEL_MAP: dict = {}


def register_hf_model(model_type: str, repo_id: str, filename: str = "model.safetensors"):
    def wrapper(remap_fn):
        _HF_MODEL_MAP[model_type] = (repo_id, filename, remap_fn)
        return remap_fn
    return wrapper


@register_hf_model("gpt2", "openai-community/gpt2")
def _remap_gpt2(state_dict, model, config):
    mapping = {
        "wte.weight": "embed.weight",
        "ln_f.weight": "ln_f.weight",
        "ln_f.bias": "ln_f.bias",
    }
    block_map = {
        "ln_1": ("ln1", False),
        "ln_2": ("ln2", False),
        "attn.c_proj": ("self_attn.out_proj", True),
        "mlp.c_fc": ("feed_forward.gate_proj", True),
        "mlp.c_proj": ("feed_forward.down_proj", True),
    }
    new_state = {}
    for hf_key, local_key in mapping.items():
        if hf_key in state_dict:
            new_state[local_key] = state_dict[hf_key]

    for i in range(config.n_layers):
        for hf_part, (local_part, is_conv1d) in block_map.items():
            hf_key = f"h.{i}.{hf_part}.weight"
            local_key = f"layers.{i}.{local_part}.weight"
            if hf_key in state_dict:
                t = state_dict[hf_key]
                if is_conv1d:
                    t = t.t()
                new_state[local_key] = t
            hf_bias = f"h.{i}.{hf_part}.bias"
            local_bias = f"layers.{i}.{local_part}.bias"
            if hf_bias in state_dict:
                new_state[local_bias] = state_dict[hf_bias]

        c_attn_w = f"h.{i}.attn.c_attn.weight"
        c_attn_b = f"h.{i}.attn.c_attn.bias"
        if c_attn_w in state_dict:
            combined_w = state_dict[c_attn_w].t()
            d = config.d_model
            new_state[f"layers.{i}.self_attn.q_proj.weight"] = combined_w[:d]
            new_state[f"layers.{i}.self_attn.k_proj.weight"] = combined_w[d:2*d]
            new_state[f"layers.{i}.self_attn.v_proj.weight"] = combined_w[2*d:]
            if c_attn_b in state_dict:
                b = state_dict[c_attn_b]
                new_state[f"layers.{i}.self_attn.q_proj.bias"] = b[:d]
                new_state[f"layers.{i}.self_attn.k_proj.bias"] = b[d:2*d]
                new_state[f"layers.{i}.self_attn.v_proj.bias"] = b[2*d:]

    if "wpe.weight" in state_dict and hasattr(model, "pe") and hasattr(model.pe, "pe"):
        new_state["pe.pe.weight"] = state_dict["wpe.weight"]

    return new_state


@register_hf_model("tinyllama", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
def _remap_llama(state_dict, model, config):
    new_state = {}
    top = {
        "model.embed_tokens.weight": "embed.weight",
        "model.norm.weight": "norm.weight",
        "lm_head.weight": "lm_head.weight",
    }
    for hf_key, local_key in top.items():
        if hf_key in state_dict:
            new_state[local_key] = state_dict[hf_key]

    hf_to_local = {
        "self_attn.q_proj": "self_attn.q_proj",
        "self_attn.k_proj": "self_attn.k_proj",
        "self_attn.v_proj": "self_attn.v_proj",
        "self_attn.o_proj": "self_attn.out_proj",
        "input_layernorm": "input_layernorm",
        "post_attention_layernorm": "post_attention_layernorm",
        "mlp.gate_proj": "mlp.gate_proj",
        "mlp.up_proj": "mlp.up_proj",
        "mlp.down_proj": "mlp.down_proj",
    }
    for i in range(config.n_layers):
        for hf_name, local_name in hf_to_local.items():
            key = f"model.layers.{i}.{hf_name}.weight"
            if key in state_dict:
                new_state[f"layers.{i}.{local_name}.weight"] = state_dict[key]

    return new_state


# ANCHOR: SafetensorsLoader
@LOADER.register("safetensors")
class SafetensorsLoader(WeightLoaderABC):
    def load(self, path, model, config):
        info = _HF_MODEL_MAP.get(config.model_type)
        if info is None:
            raise ValueError(f"No HF model mapping for '{config.model_type}'")
        repo_id, filename, remap_fn = info
        if path is None:
            path = hf_hub_download(repo_id=repo_id, filename=filename)
        state_dict = load_file(path, device="cpu")
        mapped = remap_fn(state_dict, model, config)
        missing, unexpected = model.load_state_dict(mapped, strict=False)
        return {"missing": missing, "unexpected": unexpected, "path": path}
# ENDANCHOR: SafetensorsLoader
# ─── ENDSECTION: Safetensors Loader ─────────────────────
