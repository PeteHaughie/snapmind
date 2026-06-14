# ADR 002: Per-File Model Architecture Definitions

## Status

Accepted

## Context

snapmind needs to support multiple model families (Llama, GPT-2, Mistral, DeepSeek, etc.) that differ in architecture details — attention variant, norm placement, positional encoding, activation function.

Two approaches considered:

1. **Single generic model builder** — A `build_model(config)` function that reads ModelConfig fields and assembles the right layers. Flexible but opaque — hard to understand what a given model actually looks like.
2. **Per-file model classes** — One file per architecture family (`llama.py`, `gpt2.py`). Each file explicitly defines the layer arrangement. Easy to read and debug, but requires a new file per architecture.

## Decision

Use per-file model classes. Each model file registers into the MODEL registry and explicitly composes its layers.

## Consequences

- Low barrier to understanding: open `models/llama.py` and see the full architecture
- Architecture-specific weight maps live alongside the model definition
- Adding a new architecture is a single file + registration
- Code duplication between similar architectures is acceptable when it aids clarity

## References

- HuggingFace Transformers: `modeling_llama.py`, `modeling_gpt2.py`, etc.
- nanoGPT: single file per project
