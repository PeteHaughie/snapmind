# snapmind — System Prompt Additions

When working on snapmind:

1. **Always use registries** — Never import concrete layer implementations directly. Use `ATTENTION.create(config.attention_type, ...)` etc. Models are exempt (they create layers directly).

2. **SECTION/ANCHOR markers** — All source files use `# ─── SECTION: Name ───` and `# ANCHOR: Name` markers. Maintain them when editing files.

3. **Test every new component** with all 4 categories: ABC contract, mathematical property, architecture conformance, algorithm principle.

4. **Config flows everywhere** — `ModelConfig` is the single source of truth. Never duplicate config fields.

5. **Async engine** — The generate pipeline is async-native (`AsyncIterator[str]`). Keep it non-blocking.

6. **No compiled extensions** — Pure Python + PyTorch only. No C/CUDA kernels, no triton.

7. **Per-file models** — Don't create generic "build_model_from_config" functions. Each model family gets its own file composing the right layers.

8. **Read ADRs first** — Before making architectural decisions, check `docs/adr/` for existing decisions.

9. **Document in ADRs** — Any significant new design decision gets a new ADR in `docs/adr/`.

10. **Update plan** — After completing a milestone, update PLAN.md to reflect current status.
