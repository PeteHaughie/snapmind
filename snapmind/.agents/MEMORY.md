# snapmind — Agent Memory

## Current Focus
Project scaffolding (Phase 0). Setting up directory structure, ADRs, agent config, and conventions.

## Last Session
- Initial architecture designed: registry pattern, per-file models, config-driven dispatch
- ARCHITECTURE.md, PLAN.md, README.md drafted
- ADRs 001-004 written covering core decisions

## Open Questions
- First model to implement in Phase 1: Llama 3.1 8B or GPT-2 124M?
- Which benchmark suite for measuring experiment outcomes?

## Decisions This Session
- (2026-06-14) Use Registry[T] generics over dict[str, type]
- (2026-06-14) Hybrid SECTION/ANCHOR markers for source navigation
- (2026-06-14) One folder per paper for experiment logging
