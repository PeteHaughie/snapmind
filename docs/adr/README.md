# Architecture Decision Records

This directory documents every significant architectural decision for snapmind.

## Index

| # | Title | Status |
|---|---|---|
| 001 | [Registry Pattern for Component Dispatch](001-use-registry-pattern.md) | Accepted |
| 002 | [Per-File Model Architecture Definitions](002-per-file-model-architectures.md) | Accepted |
| 003 | [PyTorch-Only Backend](003-pytorch-only-backend.md) | Accepted |
| 004 | [AI-Friendly Source Annotations](004-ai-friendly-annotations.md) | Accepted |

## Template

New ADRs follow the MADR format:

```markdown
# ADR NNN: Title

## Status

Proposed | Accepted | Deprecated | Superseded by ADR NNN

## Context

Why this decision is needed, what constraints apply, alternatives considered.

## Decision

What was decided and why.

## Consequences

What this means for the codebase — trade-offs, follow-up work, migration needs.

## References

Links to relevant discussions, PRs, papers, or other ADRs.
```
