# snapmind

A modular, pure-Python framework for local transformer inference. Every component — attention, KV cache, positional encoding, normalization, activation, sampling — is a hot-swappable plugin registered by string key.

## Why snapmind?

Existing inference engines (vLLM, llama.cpp, TensorRT-LLM, SGLang) are performance-optimized monoliths with compiled C++/CUDA kernels. Novel research ideas — new KV cache eviction policies, attention variants, positional encodings — require waiting for upstream support or hacking inside a complex codebase.

snapmind is the opposite: **everything is a plugin**. Want to test a sentence-level KV cache eviction strategy? Write one file, register it, swap it in via config. No other code changes.

## Status

Phase 4 complete (serving + polish). Lookahead Sparse Attention (FlashMemory-style) implemented and trained. 386 tests (including 49 property-based), clean lint + type check.

## Features

- **Pure Python** — no compiled extensions required. PyTorch-powered.
- **Pluggable everything** — attention (SDPA, GQA, MLA), KV cache (naive, sliding window, paged), positional encoding (learned, RoPE, none), normalization (LayerNorm, RMSNorm), activation (GELU, SiLU), sampler (greedy, temperature, top-p, top-k, mirostat), tokenizer (HF). All via string-key registries.
- **Per-file model architectures** — one file per model family (`llama.py`, `gpt2.py`, `mistral.py`).
- **Real model weights** — load from safetensors and PyTorch state dicts.
- **Async-native API** — streaming generation out of the box. OpenAI-compatible server dispatches samplers via registry (ADR 007).
- **MPS/GPU support** — auto-detects Metal, CUDA, or falls back to CPU; bf16 conversion for memory-constrained devices.
- **Paged KV cache** — block-based allocation with eviction, pure Python reimplementation of PagedAttention.
- **MLA attention** — DeepSeek-style MultiHeadLatentAttention with compressed KV cache.
- **Tiered KV cache** — fixed-size GPU pool with CPU backing, chunk-based eviction, sink+window pinning.
- **Lookahead Sparse Indexer** — dual-encoder neural memory indexer (FlashMemory-style) with decoupled training on PG-19; sigmoid-gated matching scores for KV chunk selection.
- **Property-based tests** — 49 hypothesis tests fuzzing samplers, config, registry, attention masks, and KV cache invariants.
- **Benchmark corpus** — standardised prompt corpus and runner for KV cache performance measurement (TTFT, decode throughput, memory).

## Quick Start

```bash
# Generate text from a prompt using any known model
uv run python -m snapmind.serving.cli generate --model mistral --prompt "What is attention?" --max-tokens 50

# Start an OpenAI-compatible API server
uv run python -m snapmind.serving.cli serve --model llama --port 8000

# List available models
uv run python -m snapmind.serving.cli list
```

Or in Python:

```python
import asyncio
from snapmind.models.llama import LlamaModel
from snapmind.core.config import ModelConfig
from snapmind.engine.generate import GenerateEngine
from snapmind.sampling.greedy import GreedySampler
from snapmind.tokenizer.hf import HFTokenizer

config = ModelConfig(model_type="llama", d_model=2048, n_heads=32, n_kv_heads=4, n_layers=22)
model = LlamaModel(config)
tok = HFTokenizer(model_name="tinyllama")
engine = GenerateEngine(model, tok, GreedySampler())

async def run():
    async for token in engine.generate("What is attention?", max_tokens=50):
        print(token, end="", flush=True)

asyncio.run(run())
```

## Adding a Custom Component

```python
from snapmind.core.registry import KV_CACHE
from snapmind.kv_cache.base import KVCacheABC

@KV_CACHE.register("sentence_eviction")
class SentenceEvictionKVCache(KVCacheABC):
    def store(self, layer_idx, key, value, seq_pos): ...
    def fetch(self, layer_idx): ...
    def evict(self, tokens_to_keep): ...

config = ModelConfig(kv_cache_type="sentence_eviction")
```

## Inspirations

snapmind's architecture draws from several projects:

- **FlexiTransformers** — modular transformer construction with a pluggable positional encoding registry (`register_pe`). The registry-and-ABC pattern used throughout snapmind is a generalization of this design.
- **HuggingFace Transformers** — per-file model definitions and the `register_model` pattern. snapmind adopts the same approach of one file per architecture family.
- **Karpathy's llama2.py / nanoGPT** — the philosophy that understanding inference shouldn't require navigating a 200K-line codebase. snapmind aims to keep its core ~5K lines.
- **vLLM PagedAttention** — snapmind's `PagedKVCache` is an independent reimplementation of the same idea, but as a swappable plugin rather than a hard-coded kernel.
- **Modular (MAX + Mojo)** — the [Modular Platform](https://github.com/modular/modular) demonstrated three patterns that influenced snapmind's evolution: (1) a `SupportedArchitecture` dataclass pattern that separates architecture metadata from model implementation, keeping registration concerns cleanly decoupled; (2) a composite config model that auto-routes CLI flags to sub-configs (sampling, runtime, profiling); and (3) per-architecture weight format adapters for loading from multiple checkpoint formats.
- **Comment Anchors (VSCode)** — SECTION/ANCHOR markers for AI-navigable source code, inspired by VSCode comment anchor extensions.
- **CLAUDE.md / MADR** — Memory files, agent configuration, and Architecture Decision Records for persistent agent context and structured decision logging.

## AI Collaboration

snapmind is designed for AI-assisted engineering. The codebase uses structured markers and conventions that make it easy for coding agents to navigate:

- **SECTION/ENDSECTION markers** — agents can `grep` for `SECTION: Attention` to find relevant code without reading entire files
- **ANCHOR/ENDANCHOR markers** — key decision points (class defs, plugin hooks) are explicitly tagged
- **`docs/adr/`** — Architecture Decision Records prevent contradictory decisions across sessions
- **`docs/experiments/`** — Every paper-based implementation logged with status, results, and rejection reason
- **`docs/benchmarks/`** — Standardised prompt corpus, runner, and per-model results for KV cache performance measurement
- **`.agents/`** — Tool-specific instructions for AI coding agents
- **`MEMORY.md`** — Persistent session context so agents don't lose focus between invocations

## Project Structure

```
snapmind/
├── .agents/       # AI agent configuration and memory
├── core/          # Foundation: registries, configs
├── models/        # One file per architecture (llama.py, gpt2.py, ...)
├── layers/        # Primitive building blocks (attention, PE, norm, activations)
├── kv_cache/      # Pluggable KV cache strategies
├── tokenizer/     # Pluggable tokenization
├── sampling/      # Pluggable sampling strategies
├── engine/        # Prefill, decode, generate pipeline
├── loaders/       # Weight format loaders
├── serving/       # CLI and HTTP server
├── docs/
│   ├── adr/       # Architecture Decision Records
│   ├── experiments/  # One folder per tested paper/idea
│   └── benchmarks/   # Corpus, runner, and results for KV cache measurement
└── MEMORY.md      # Session-persistent agent context
```

## License

MIT
