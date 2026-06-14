# snapmind MEMORY

## Project
Modular pure-Python transformer inference framework. Every component is a hot-swappable plugin.

## Current Focus
Phase 0 — Project scaffolding. Setting up agent config, ADRs, experiment tracking conventions.

## Next Phase
Phase 1 — Foundation + Layers. Build core/registry.py, core/config.py, all layer ABCs and defaults, first model (llama.py).

## Recent Decisions
- (2026-06-14) `Registry[T]` pattern for all component dispatch
- (2026-06-14) Per-file model architectures (llama.py, gpt2.py, etc.)
- (2026-06-14) PyTorch-only backend (for now)
- (2026-06-14) SECTION/ANCHOR markers in source files for AI navigation
- (2026-06-14) One folder per paper for experiment logging

## Open Questions
- First model for Phase 1: GPT-2 124M (small, fast iteration) or Llama 3.1 8B (real-world)?
- What benchmark corpus for measuring KV cache experiments?
- Should we support .gguf weights or only safetensors for now?

## Working Context
- All docs are in project root: ARCHITECTURE.md, PLAN.md, README.md
- ADRs: docs/adr/
- Experiments: docs/experiments/
- Agent config: .agents/ and snapmind/.agents/
