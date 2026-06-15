# snapmind — Project-Level Agent Instructions

## Overview

snapmind is a modular pure-Python transformer inference framework. Every component is a hot-swappable plugin registered by string key. The framework never imports concrete implementations — it dispatches through registries.

## Key Conventions

- **Registry pattern**: Components register via `@REGISTRY.register("key")` decorator. Dispatch via `REGISTRY.create("key", **kwargs)`.
- **Per-file models**: One file per architecture (gpt2.py, llama.py, etc.).
- **ABCs**: Every component type has an ABC in a `base.py` file proving its contract.
- **SECTION/ANCHOR**: Source code uses `# ─── SECTION: ───` and `# ANCHOR:` markers for AI navigation.
- **ADRs**: All significant decisions in `docs/adr/`.
- **Experiments**: Paper-based implementations logged in `docs/experiments/<name>/`.

## Codebase Structure

```
snapmind/
├── core/           # Registry, Config
├── layers/         # Primitive building blocks (attention, PE, norm, activation, FFN)
│   ├── attention/
│   ├── positional/
│   ├── normalization/
│   └── activation/
├── models/         # Full model architectures (gpt2.py, llama.py)
├── kv_cache/       # KV cache strategies
├── tokenizer/      # Tokenization
├── sampling/       # Token selection strategies
├── engine/         # Prefill, decode, generate pipeline
├── loaders/        # Weight format loaders (safetensors, pytorch)
├── serving/        # CLI and HTTP server
├── tests/          # All tests (206+ tests across 4 categories)
└── .agents/        # Framework-specific agent config
```

## Post-Cycle Requirements

After each development cycle, before wrapping up:
- **Update the README** at `/README.md` to reflect the current state: bump status if applicable, add new features, update quick-start examples to match current APIs, and verify all references are accurate.

## Testing

```bash
uv run pytest snapmind/tests/ -v           # All tests
uv run pytest snapmind/tests/ -k "llama"   # Filter by model
```

Test categories: ABC contract, mathematical property, architecture conformance, algorithm principle.

## Environment

- Package manager: `uv` (uv sync, uv run, uv add)
- Python: 3.13 (set in .python-version)
- Key deps: torch, huggingface_hub, safetensors, tiktoken
- HF_TOKEN env var must be set for HuggingFace model access
