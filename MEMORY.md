# snapmind MEMORY

## Project
Modular pure-Python transformer inference framework. Every component is a hot-swappable plugin.

## Current Focus
Phase 0 complete. Phase 1-3 complete. 206 tests passing. 2 models: GPT-2 and Llama.

## Recent Decisions
- (2026-06-14) Registry[T] pattern for all component dispatch
- (2026-06-14) Per-file model architectures
- (2026-06-14) PyTorch-only backend
- (2026-06-14) SECTION/ANCHOR markers in source files for AI navigation
- (2026-06-14) One folder per paper for experiment logging
- (2026-06-14) Llama model: separate LlamaTransformerBlock, apply_to_qk on PE ABC, GQA with RoPE injection, GatedFeedForward as separate layer
- (2026-06-14) Agent config files in .agents/ directories

## Open Questions
- Benchmark corpus for KV cache performance measurement
- Integration test for real Llama weights
- Should MODEL registry be used for all models including GPT-2?

## Working Context
- Project root: /Users/petehaughie/Projects/python-inference-framework
- Package: snapmind
- Docs: ARCHITECTURE.md, PLAN.md, docs/adr/, docs/experiments/
- Agent config: .agents/ and snapmind/.agents/
