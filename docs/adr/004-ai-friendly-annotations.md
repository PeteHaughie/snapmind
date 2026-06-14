# ADR 004: AI-Friendly Source Annotations

## Status

Accepted

## Context

snapmind expects that most engineering work will be performed by AI coding agents (Claude Code, GitHub Copilot, etc.) rather than humans editing files directly. Source code must be navigable programmatically by LLMs with limited context windows.

Requirements:
- Agents must be able to locate relevant code sections without reading entire files
- File structure must be parseable by simple regex/pattern matching
- Experiment history and paper provenance must be discoverable

## Decision

1. **SECTION/ENDSECTION markers** for large block boundaries (modules, classes)

    ```python
    # ─── SECTION: Attention Base ───────────────────────────
    ...
    # ─── ENDSECTION: Attention Base ────────────────────────
    ```

2. **ANCHOR/ENDANCHOR markers** for key decision points inside sections

    ```python
    # ANCHOR: class AttentionABC
    class AttentionABC(ABC): ...
    # ENDANCHOR: class AttentionABC
    ```

3. **One folder per paper** in `docs/experiments/<name>/` with structured README
4. **ADRs in `docs/adr/`** capturing every significant decision with rationale
5. **MEMORY.md** at project root for session-persistent agent context
6. **`.agents/` directories** with AGENTS.md + SYSTEM.md for tool-specific instructions

## Consequences

- Source files have more comment lines. Accepted trade-off for AI navigability.
- SECTION markers make files scannable — agents can `grep` for `SECTION: <topic>` to find relevant code
- Experiment folders create a searchable paper trail: what was tried, what worked, what didn't, and why
- ADRs prevent agents from making contradictory decisions across sessions

## References

- VSCode Comment Anchors extension
- Anthropic's CLAUDE.md conventions
- MADR (Markdown Architectural Decision Records)
