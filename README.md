# snapmind

A modular, pure-Python framework for local transformer inference. Every component — attention, KV cache, positional encoding, normalization, activation, sampling — is a hot-swappable plugin registered by string key.

## Why snapmind?

Existing inference engines (vLLM, llama.cpp, TensorRT-LLM, SGLang) are performance-optimized monoliths with compiled C++/CUDA kernels. Novel research ideas — new KV cache eviction policies, attention variants, positional encodings — require waiting for upstream support or hacking inside a complex codebase.

snapmind is the opposite: **everything is a plugin**. Want to test a sentence-level KV cache eviction strategy? Write one file, register it, swap it in via config. No other code changes.

## Status

Early development — Phase 1 of 4. Not yet ready for production use.

## Features

- **Pure Python** — no compiled extensions required. PyTorch-powered.
- **Pluggable everything** — attention, KV cache, positional encoding, normalization, activation, sampler, tokenizer. All via string-key registries.
- **Per-file model architectures** — one file per model family (`llama.py`, `gpt2.py`, `mistral.py`).
- **Real model weights** — load from safetensors and PyTorch state dicts.
- **Async-native API** — streaming generation out of the box.

## Quick Start

```python
from snapmind import load_model, generate
from snapmind.core.config import ModelConfig

config = ModelConfig(model_type="llama", kv_cache_type="naive")
model = load_model(config, weights="path/to/model.safetensors")

for token in generate(model, tokenizer, "What is attention?"):
    print(token, end="", flush=True)
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
- **Comment Anchors (VSCode)** — SECTION/ANCHOR markers for AI-navigable source code, inspired by VSCode comment anchor extensions.
- **CLAUDE.md / MADR** — Memory files, agent configuration, and Architecture Decision Records for persistent agent context and structured decision logging.

## AI Collaboration

snapmind is designed for AI-assisted engineering. The codebase uses structured markers and conventions that make it easy for coding agents to navigate:

- **SECTION/ENDSECTION markers** — agents can `grep` for `SECTION: Attention` to find relevant code without reading entire files
- **ANCHOR/ENDANCHOR markers** — key decision points (class defs, plugin hooks) are explicitly tagged
- **`docs/adr/`** — Architecture Decision Records prevent contradictory decisions across sessions
- **`docs/experiments/`** — Every paper-based implementation logged with status, results, and rejection reason
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
│   └── experiments/  # One folder per tested paper/idea
└── MEMORY.md      # Session-persistent agent context
```

## License

MIT
