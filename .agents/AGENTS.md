# snapmind Agent Instructions

This project is a modular pure-Python transformer inference framework. Every component is a hot-swappable plugin.

## Key Navigation Points

- **ARCHITECTURE.md** — Full architecture docs (registry system, component ownership, data flow)
- **PLAN.md** — Implementation phases and current delivery status
- **docs/adr/** — Architecture Decision Records for every significant decision
- **docs/experiments/** — One folder per tested paper/idea, with status and results
- **MEMORY.md** — Session-persistent context (current focus, open questions)

## Conventions

- All component registries are in `snapmind/core/registry.py` as singleton `Registry` instances
- Every ABC is registered as `snapmind/layers/<component>/base.py`
- Model architectures live one per file in `snapmind/models/`
- Config is a single `ModelConfig` dataclass — change component selection by string key
- No compiled extensions — pure Python, PyTorch only

## Anchor Markers

Source files use `SECTION`/`ENDSECTION` for block boundaries and `ANCHOR`/`ENDANCHOR` for key decision points:

```python
# ─── SECTION: Attention Base ───────────────────────────
# ANCHOR: class AttentionABC
class AttentionABC(ABC): ...
# ENDANCHOR: class AttentionABC
# ─── ENDSECTION: Attention Base ────────────────────────
```

## When Implementing a New Component

1. Check `docs/adr/` for any relevant prior decisions
2. Check `docs/experiments/` if this has been tried before
3. Check `snapmind/layers/<component>/base.py` for the ABC contract
4. Register the component: `@REGISTRY.register("key")`
5. Log the experiment paper + results to `docs/experiments/<name>/`
