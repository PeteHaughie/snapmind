# ADR 003: PyTorch-Only Backend

## Status

Accepted

## Context

snapmind needs a tensor backend for GPU/CPU operations. Options:

1. **PyTorch only** — Use `torch` directly throughout. Simplest, fastest development.
2. **Abstract tensor backend** — Define a tensor ops ABC with PyTorch + MLX + NumPy implementations. Maximum portability.
3. **PyTorch + MLX** — Support the two most common backends for local inference.

## Decision

Start with PyTorch only. Abstracting the tensor backend adds significant complexity for no immediate benefit. The abstraction can be added later if MLX or other backends are demanded.

## Consequences

- Faster initial development
- Users must have PyTorch installed (de facto standard anyway)
- No lock-in to PyTorch-specific APIs — core code uses standard PyTorch; migration to an abstraction layer later is feasible
- Mac MPS backend available via PyTorch MPS as an intermediate step

## References

- PyTorch is the most widely used DL framework in research, making it the natural first target
