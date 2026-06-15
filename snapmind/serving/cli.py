# ─── SECTION: CLI ────────────────────────────────────────
from __future__ import annotations

import argparse
import sys

import torch.nn as nn

import snapmind.models.gpt2  # noqa: F401 — triggers ARCHITECTURE registration
import snapmind.models.llama  # noqa: F401
import snapmind.models.mistral  # noqa: F401
from snapmind.core.architecture import ARCHITECTURE
from snapmind.core.config import ModelConfig


def _get_model_names() -> list[str]:
    return sorted(ARCHITECTURE.list())


# ANCHOR: build_model
def build_model(model_name: str, device: str = "auto") -> tuple[nn.Module, ModelConfig]:
    arch = ARCHITECTURE.get(model_name)
    cfg = ModelConfig(**arch.default_config)
    model: nn.Module = arch.model_cls(cfg)

    import torch

    for p in model.parameters():
        p.data = p.data.to(torch.bfloat16)

    load_weights(model, model_name)
    from snapmind.core.device import resolve_device

    target = resolve_device(device)
    model = model.to(target)
    return model, cfg


# ENDANCHOR: build_model


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


# ENDANCHOR: load_weights


# ANCHOR: cmd_generate
def cmd_generate(args: argparse.Namespace):
    import asyncio

    from snapmind.core.config import SamplingConfig
    from snapmind.engine.generate import GenerateEngine
    from snapmind.sampling.greedy import GreedySampler
    from snapmind.tokenizer.hf import HFTokenizer

    model, _ = build_model(args.model, device=args.device)
    tok = HFTokenizer(model_name=args.model)
    sampler = GreedySampler()
    samp_cfg = SamplingConfig(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_k=args.top_k,
        top_p=args.top_p,
    )
    engine = GenerateEngine(model, tok, sampler)

    async def run():
        tokens = []
        async for token in engine.generate(
            args.prompt,
            max_tokens=samp_cfg.max_tokens,
            temperature=samp_cfg.temperature,
            top_k=samp_cfg.top_k,
            top_p=samp_cfg.top_p,
        ):
            tokens.append(token)
        return "".join(tokens)

    output = asyncio.run(run())
    print(output)


# ENDANCHOR: cmd_generate


# ANCHOR: cmd_serve
def cmd_serve(args: argparse.Namespace):
    import uvicorn

    from snapmind.serving.openai_api import create_app

    model, cfg = build_model(args.model, device=args.device)
    app = create_app(model, cfg, args.model)
    uvicorn.run(app, host=args.host, port=args.port)


# ENDANCHOR: cmd_serve


# ANCHOR: cmd_list
def cmd_list(args: argparse.Namespace):
    print("Available models:")
    for name in _get_model_names():
        arch = ARCHITECTURE.get(name)
        cfg = arch.default_config
        kv = cfg.get("n_kv_heads", cfg.get("n_heads", "?"))
        print(
            f"  {name}: {cfg.get('n_layers', '?')} layers, {cfg.get('n_heads', '?')} heads,"
            f" {kv} KV heads, {cfg.get('d_model', '?')} dim, {cfg.get('vocab_size', '?')} vocab"
        )


# ENDANCHOR: cmd_list


# ANCHOR: main
def main() -> None:
    parser = argparse.ArgumentParser(prog="snapmind", description="snapmind — transformer inference framework")
    sub = parser.add_subparsers(dest="command", required=True)

    model_names = _get_model_names()

    p_serve = sub.add_parser("serve", help="Start an OpenAI-compatible API server")
    p_serve.add_argument("--model", default="gpt2", choices=model_names)
    p_serve.add_argument("--device", default="auto", help="Device: auto, cpu, mps, cuda")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.set_defaults(func=cmd_serve)

    p_gen = sub.add_parser("generate", help="Generate text from a prompt")
    p_gen.add_argument("--model", default="gpt2", choices=model_names)
    p_gen.add_argument("--device", default="auto", help="Device: auto, cpu, mps, cuda")
    p_gen.add_argument("--prompt", default="Hello")
    p_gen.add_argument("--max-tokens", type=int, default=50)
    p_gen.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    p_gen.add_argument("--top-k", type=int, default=None, help="Top-k sampling")
    p_gen.add_argument("--top-p", type=float, default=None, help="Top-p (nucleus) sampling")
    p_gen.set_defaults(func=cmd_generate)

    p_list = sub.add_parser("list", help="List known model architectures")
    p_list.set_defaults(func=cmd_list)

    parsed = parser.parse_args()
    parsed.func(parsed)


# ENDANCHOR: main

if __name__ == "__main__":
    main()
# ─── ENDSECTION: CLI ─────────────────────────────────────
