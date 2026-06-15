# snapmind MEMORY

## Project
Modular pure-Python transformer inference framework. Every component is a hot-swappable plugin.

## Current Focus
Phases 0-3 complete. Phase 4 in progress. 313 tests passing. 4 models: GPT-2, Llama, Mistral, Ministral 3 3B. 10 attention/KV/sampler plugins.

## Recent Decisions
- (2026-06-14) Registry[T] pattern for all component dispatch
- (2026-06-14) Per-file model architectures
- (2026-06-14) PyTorch-only backend
- (2026-06-14) SECTION/ANCHOR markers in source files for AI navigation
- (2026-06-14) One folder per paper for experiment logging
- (2026-06-14) Llama model: separate LlamaTransformerBlock, apply_to_qk on PE ABC, GQA with RoPE injection, GatedFeedForward as separate layer
- (2026-06-14) Agent config files in .agents/ directories
- (2026-06-14) Ministral 3 3B: head_dim param, tie_word_embeddings, consolidated.safetensors format
- (2026-06-14) MPS device support: resolve_device(), bf16 conversion before weight load
- (2026-06-14) PagedKVCache: block-based allocation, free-list management, partial-block eviction
- (2026-06-14) Top-K and Mirostat samplers
- (2026-06-14) MLA (MultiHeadLatentAttention): DeepSeek-style compressed KV cache
- (2026-06-15) ADR 007: Registry-native OpenAI API server (Proposed)
- (2026-06-15) Modular platform research: SupportedArchitecture pattern, composite config, weight adapters

## Open Questions
- Benchmark corpus for KV cache performance measurement
- Integration test for real Llama weights
- Should MODEL registry be used for all models including GPT-2?
- Registry-native OpenAI API server implementation
- SupportedArchitecture dataclass pattern for model registry
- Composite config model for CLI/API

## Working Context
- Project root: /Users/petehaughie/Projects/python-inference-framework
- Package: snapmind
- Docs: ARCHITECTURE.md, PLAN.md, docs/adr/, docs/experiments/
- Agent config: .agents/ and snapmind/.agents/
