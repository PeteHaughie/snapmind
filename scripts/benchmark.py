#!/usr/bin/env python3
"""
Benchmark runner for KV cache performance measurement.

Usage:
    python scripts/benchmark.py                              # GPT-2 tiny, all prompts
    python scripts/benchmark.py --model tinyllama            # TinyLlama arch, random weights
    python scripts/benchmark.py --seq-lens 16 64 256         # Custom lengths
    python scripts/benchmark.py --output results/run1.tsv    # Save to file
    python scripts/benchmark.py --warmup 3 --samples 5       # More precise timing
"""

import argparse
import json
import pathlib
import sys
import time

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import snapmind.models.gpt2  # noqa: F401
import snapmind.models.llama  # noqa: F401
import snapmind.models.mistral  # noqa: F401
from snapmind.core.architecture import ARCHITECTURE
from snapmind.core.config import ModelConfig
from snapmind.engine.decode import decode_step
from snapmind.engine.prefill import prefill
from snapmind.sampling.greedy import GreedySampler
from snapmind.tokenizer.hf import HFTokenizer

CORPUS_PATH = pathlib.Path(__file__).resolve().parent.parent / "docs" / "benchmarks" / "corpus.json"


def load_corpus() -> list[dict]:
    with open(CORPUS_PATH) as f:
        data = json.load(f)
    return data["entries"]


def build_model(model_name: str) -> tuple[torch.nn.Module, ModelConfig]:
    arch = ARCHITECTURE.get(model_name)
    cfg = ModelConfig(**arch.default_config)
    model: torch.nn.Module = arch.model_cls(cfg)
    for p in model.parameters():
        p.data = p.data.to(torch.bfloat16)
    model.eval()
    return model, cfg


def measure_prefill(
    model: torch.nn.Module, input_ids: torch.Tensor
) -> tuple[torch.Tensor, float]:
    kv_cache = {i: {"k": None, "v": None} for i in range(model.config.n_layers)}
    t0 = time.perf_counter()
    logits, ttft = prefill(model, input_ids, kv_cache)
    t1 = time.perf_counter()
    return logits, ttft, kv_cache, (t1 - t0)


def measure_decode(
    model: torch.nn.Module,
    kv_cache: dict,
    first_token_id: int,
    sampler: GreedySampler,
    num_steps: int,
    position_offset: int,
) -> float:
    token_id = first_token_id
    device = next(model.parameters()).device
    t0 = time.perf_counter()
    for step in range(num_steps):
        pos = position_offset + step
        token_id = decode_step(
            model,
            token_id,
            kv_cache,
            sampler,
            temperature=1.0,
            position_ids=torch.tensor([pos], device=device),
        )
    t1 = time.perf_counter()
    return (t1 - t0) / num_steps


def measure_cache_memory(kv_cache: dict) -> float:
    total_bytes = 0
    for layer_cache in kv_cache.values():
        k = layer_cache.get("k")
        v = layer_cache.get("v")
        if k is not None:
            total_bytes += k.element_size() * k.numel()
        if v is not None:
            total_bytes += v.element_size() * v.numel()
    return total_bytes / (1024 * 1024)


def run_benchmark(
    model_name: str,
    seq_lens: list[int] | None,
    warmup: int,
    samples: int,
    decode_steps: int,
) -> list[dict]:
    print(f"Building model: {model_name} ...", file=sys.stderr)
    model, cfg = build_model(model_name)
    sampler = GreedySampler()

    try:
        tokenizer = HFTokenizer(model_name=model_name)
    except Exception:
        tokenizer = HFTokenizer(model_name="gpt2")

    device = next(model.parameters()).device
    print(f"  Device: {device}, Parameters: {sum(p.numel() for p in model.parameters()):,}", file=sys.stderr)

    corpus = load_corpus()
    results = []

    for entry in corpus:
        prompt = entry["prompt"]
        input_ids = tokenizer.encode(prompt)
        if isinstance(input_ids, list):
            seq_len = len(input_ids)
        else:
            seq_len = len(input_ids[0]) if hasattr(input_ids, "__len__") else 16

        if seq_lens is not None and seq_len not in seq_lens:
            continue

        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

        for _ in range(warmup):
            _, _, kv_cache, _ = measure_prefill(model, input_tensor)
            first_token = int(sampler.sample(kv_cache[0]["k"][:, :, -1, :].mean(dim=-2).argmax(dim=-1)).item())
            measure_decode(model, kv_cache, first_token, sampler, decode_steps, seq_len)

        prefills = []
        decodes = []
        memories = []
        for _ in range(samples):
            logits, ttft, kv_cache, wall = measure_prefill(model, input_tensor)
            prefills.append(ttft * 1000)

            first_token = int(sampler.sample(logits).item())
            sec_per_step = measure_decode(model, kv_cache, first_token, sampler, decode_steps, seq_len)
            decodes.append(1.0 / sec_per_step)

            memories.append(measure_cache_memory(kv_cache))

        avg_prefill = sum(prefills) / len(prefills)
        avg_decode = sum(decodes) / len(decodes)
        avg_mem = sum(memories) / len(memories)

        result = {
            "model": model_name,
            "prompt": entry["name"],
            "seq_len": seq_len,
            "prefill_ttft_ms": f"{avg_prefill:.2f}",
            "decode_tok_per_sec": f"{avg_decode:.1f}",
            "cache_memory_mb": f"{avg_mem:.2f}",
            "d_model": cfg.d_model,
            "n_layers": cfg.n_layers,
            "n_heads": cfg.n_heads,
            "n_kv_heads": cfg.n_kv_heads,
        }
        results.append(result)
        status = (
            f"  {entry['name']:6s}  len={seq_len:<5d}  "
            f"prefill={avg_prefill:8.2f}ms  "
            f"decode={avg_decode:8.1f} tok/s  "
            f"cache={avg_mem:7.2f}MB"
        )
        print(status, file=sys.stderr)

    return results


def write_tsv(results: list[dict], file):
    if not results:
        return
    headers = list(results[0].keys())
    print("\t".join(headers), file=file)
    for r in results:
        print("\t".join(str(r[h]) for h in headers), file=file)


def main():
    parser = argparse.ArgumentParser(description="KV cache benchmark runner")
    parser.add_argument("--model", default="gpt2", choices=sorted(ARCHITECTURE.list()))
    parser.add_argument("--seq-lens", type=int, nargs="*", default=None)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--decode-steps", type=int, default=10)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    results = run_benchmark(
        model_name=args.model,
        seq_lens=args.seq_lens,
        warmup=args.warmup,
        samples=args.samples,
        decode_steps=args.decode_steps,
    )

    if args.output:
        out_path = pathlib.Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            write_tsv(results, f)
        print(f"\nResults written to {out_path}", file=sys.stderr)
    else:
        write_tsv(results, sys.stdout)


if __name__ == "__main__":
    main()
