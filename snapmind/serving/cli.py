# ─── SECTION: CLI ────────────────────────────────────────
from __future__ import annotations

import argparse
import sys
from typing import Any

import torch.nn as nn

from snapmind.core.config import ModelConfig

KNOWN_MODELS: dict[str, dict[str, Any]] = {
    "gpt2": {"d_model": 768, "n_heads": 12, "n_layers": 12, "vocab_size": 50257, "max_seq_len": 1024},
    "tinyllama": {
        "d_model": 2048,
        "n_heads": 32,
        "n_kv_heads": 4,
        "n_layers": 22,
        "vocab_size": 32000,
        "max_seq_len": 2048,
        "d_ff": 5632,
        "norm_eps": 1e-05,
        "rope_theta": 10000.0,
    },
    "llama": {
        "d_model": 4096,
        "n_heads": 32,
        "n_kv_heads": 8,
        "n_layers": 32,
        "vocab_size": 32000,
        "max_seq_len": 8192,
    },
    "mistral": {
        "d_model": 4096,
        "n_heads": 32,
        "n_kv_heads": 8,
        "n_layers": 32,
        "vocab_size": 32000,
        "max_seq_len": 8192,
        "d_ff": 14336,
        "norm_eps": 1e-05,
        "rope_theta": 10000.0,
        "window_size": 4096,
    },
}


# ANCHOR: build_model
def build_model(model_name: str) -> tuple[nn.Module, ModelConfig]:
    from snapmind.core.config import ModelConfig

    spec = KNOWN_MODELS.get(model_name)
    if spec is None:
        print(f"Error: unknown model '{model_name}'. Known: {', '.join(KNOWN_MODELS)}", file=sys.stderr)
        sys.exit(1)

    cfg = ModelConfig(
        model_type="gpt2" if model_name == "gpt2" else model_name,
        d_model=spec["d_model"],
        n_heads=spec["n_heads"],
        n_kv_heads=spec.get("n_kv_heads", spec["n_heads"]),
        n_layers=spec["n_layers"],
        vocab_size=spec["vocab_size"],
        max_seq_len=spec["max_seq_len"],
        norm_eps=spec.get("norm_eps", 1e-5),
        d_ff=spec.get("d_ff", spec["d_model"] * 4),
        rope_theta=spec.get("rope_theta", 10000.0),
    )
    if model_name == "gpt2":
        from snapmind.models.gpt2 import GPT2Model

        model: nn.Module = GPT2Model(cfg)
    elif model_name == "mistral":
        from snapmind.models.mistral import MistralModel

        model = MistralModel(cfg)
    else:
        from snapmind.models.llama import LlamaModel

        model = LlamaModel(cfg)

    load_weights(model, model_name)
    return model, cfg


# ANCHOR: load_weights
def load_weights(model: nn.Module, model_name: str) -> None:
    from snapmind.core.registry import LOADER

    try:
        loader = LOADER.create("safetensors")
        result = loader.load(None, model, model.config)
        if result["missing"]:
            missing_str = ", ".join(list(result["missing"])[:5])
            print(f"  Warning: {len(result['missing'])} missing keys: {missing_str}...", file=sys.stderr)
    except Exception as e:
        print(f"  Warning: weight loading failed ({e}). Using random weights.", file=sys.stderr)


# ANCHOR: cmd_generate
def cmd_generate(args: argparse.Namespace):
    import asyncio

    from snapmind.engine.generate import GenerateEngine
    from snapmind.sampling.greedy import GreedySampler
    from snapmind.tokenizer.hf import HFTokenizer

    model, _ = build_model(args.model)
    tok = HFTokenizer(model_name=args.model)
    sampler = GreedySampler()
    engine = GenerateEngine(model, tok, sampler)

    async def run():
        tokens = []
        async for token in engine.generate(args.prompt, max_tokens=args.max_tokens):
            tokens.append(token)
        return "".join(tokens)

    output = asyncio.run(run())
    print(output)


# ENDANCHOR: cmd_generate


# ANCHOR: cmd_serve
def cmd_serve(args: argparse.Namespace):
    import uvicorn

    from snapmind.serving.openai_api import create_app

    model, cfg = build_model(args.model)
    app = create_app(model, cfg, args.model)
    uvicorn.run(app, host=args.host, port=args.port)


# ENDANCHOR: cmd_serve


# ANCHOR: cmd_list
def cmd_list(args: argparse.Namespace):
    print("Available models:")
    for name, spec in KNOWN_MODELS.items():
        kv = spec.get("n_kv_heads", spec["n_heads"])
        print(
            f"  {name}: {spec['n_layers']} layers, {spec['n_heads']} heads, {kv} KV heads, {spec['d_model']} dim, {spec['vocab_size']} vocab"
        )


# ENDANCHOR: cmd_list


# ANCHOR: main
def main() -> None:
    parser = argparse.ArgumentParser(prog="snapmind", description="snapmind — transformer inference framework")
    sub = parser.add_subparsers(dest="command", required=True)

    p_serve = sub.add_parser("serve", help="Start an OpenAI-compatible API server")
    p_serve.add_argument("--model", default="gpt2", choices=list(KNOWN_MODELS))
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.set_defaults(func=cmd_serve)

    p_gen = sub.add_parser("generate", help="Generate text from a prompt")
    p_gen.add_argument("--model", default="gpt2", choices=list(KNOWN_MODELS))
    p_gen.add_argument("--prompt", default="Hello")
    p_gen.add_argument("--max-tokens", type=int, default=50)
    p_gen.set_defaults(func=cmd_generate)

    p_list = sub.add_parser("list", help="List known model architectures")
    p_list.set_defaults(func=cmd_list)

    parsed = parser.parse_args()
    parsed.func(parsed)


# ENDANCHOR: main

if __name__ == "__main__":
    main()
# ─── ENDSECTION: CLI ─────────────────────────────────────
