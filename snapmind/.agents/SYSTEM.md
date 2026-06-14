# System Prompt Additions for snapmind

When working with AI coding tools on this project, include the following context:

## Project Identity

```
You are contributing to snapmind — a modular pure-Python transformer inference framework.
Every component is a plugin registered by string key. Architecture decisions are documented in docs/adr/.
Source files use SECTION/ENDSECTION and ANCHOR/ENDANCHOR markers for navigation.
```

## Coding Rules

1. Every new pluggable component needs: ABC in `layers/<type>/base.py`, default impl, registry registration, ARCHITECTURE.md update
2. No compiled extensions. Pure Python + PyTorch only.
3. Model architectures go in `models/`, one file per family
4. Keep core framework under ~5K lines. Complexity is a cost.
5. All experiments based on papers must log to `docs/experiments/<name>/`
6. Before implementing something new, check `docs/experiments/` and `docs/adr/` first
7. Every significant decision needs an ADR

## File Template for a New Component

```python
# ─── SECTION: <Component Name> ───────────────────────
# ANCHOR: class <ClassName>
from snapmind.core.registry import <REGISTRY>

@<REGISTRY>.register("<key>")
class <ClassName>(<BaseClass>):
    def __init__(self, ...):
        ...

    def forward(self, ...):
        ...
# ENDANCHOR: class <ClassName>
# ─── ENDSECTION: <Component Name> ────────────────────
```
