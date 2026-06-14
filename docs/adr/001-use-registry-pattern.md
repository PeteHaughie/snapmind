# ADR 001: Registry Pattern for Component Dispatch

## Status

Accepted

## Context

Every component in snapmind (attention, KV cache, positional encoding, normalization, activation, sampler, tokenizer, loader, model) needs to be independently swappable. The framework must never import concrete component classes — it must dispatch through an abstraction.

Three approaches were considered:

1. **Subclassing** — Users subclass a base model and override methods. Fragile, violates Open/Closed principle, conflicts with composing multiple swappable parts.
2. **Dependency injection** — Components passed as constructor arguments. Clean but requires users to wire everything manually. High boilerplate.
3. **Registry dispatch** — Components register themselves by string key; framework dispatches via `registry.create(key)`. Users change a config string to swap. Zero wiring.

## Decision

Use a generic `Registry[T]` class with decorator-based registration. One singleton instance per component type. Framework code calls `REGISTRY.create(config.key, **kwargs)` and never knows the concrete class.

## Consequences

- Users add new components without modifying framework code — just write a file and register
- Introspection is free: `registry.list()` for debugging, tooling, and AI navigation
- Type safety via generics: `Registry[AttentionABC]` only creates `AttentionABC` subtypes
- Slight indirection cost on creation (dict lookup), negligible at inference time

## References

- FlexiTransformers `register_pe()` system
- HuggingFace `register_model()` / `AutoModel`
