# snapmind Framework — Agent Instructions

This directory contains configuration for AI agents contributing to the snapmind framework source code.

## Files

- `AGENTS.md` — This file. Agent instructions for framework contributions.
- `SYSTEM.md` — System prompt additions for LLM tooling when working on this codebase.
- `MEMORY.md` — Persistent session memory shared across agent sessions.

## Code Navigation

Source files use section markers for easy parsing. Every file has:

```
# ─── SECTION: <Module Name> ───────────────────────
...
# ─── ENDSECTION: <Module Name> ────────────────────
```

Key decision points inside sections use anchor markers:

```
# ANCHOR: <name>
...
# ENDANCHOR: <name>
```

## Registry Locations

All registries are instantiated in `snapmind/core/registry.py`. Import the ones you need:

```python
from snapmind.core.registry import (
    ATTENTION, PE, NORM, ACTIVATION,
    KV_CACHE, SAMPLER, TOKENIZER, LOADER, MODEL,
)
```

## Experiment Logging

Any implementation based on a paper must be logged in `docs/experiments/<name>/` with:
- Paper reference (arXiv/HF URL)
- Status (accepted / rejected / experimental)
- Rejection reason if applicable
- Results
