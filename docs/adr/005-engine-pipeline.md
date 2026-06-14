# ADR 005: Engine Pipeline Design

## Status

Accepted

## Context

The engine must orchestrate the full text generation pipeline: tokenize prompt, run the model forward pass, sample tokens, manage KV cache lifetime, stream output, and stop at the right time. Every step (prefill, decode, sampling) should be independently replaceable for future innovation (speculative decoding, parallel decoding, KV cache compression).

Three decomposition strategies were considered:

1. **Monolithic `generate()`** — Everything in one function with inline steps. Simple but untestable, and every future optimization requires touching the same function.
2. **Callbacks** — Prefill/decode steps passed as higher-order functions. Flexible but hard to configure, and error messages cross a function boundary.
3. **Separate modules per step** — `prefill.py`, `decode.py`, `generate.py` as independent functions, plus a `GenerateEngine` class that composes them. Each function is testable in isolation, and a new decode strategy doesn't touch prefill.

## Decision

Three separate modules, each with a single public function, composed by a class-based `GenerateEngine`:

- **`engine/prefill.py`** — `prefill(model, tokens, kv_cache) → (logits, ttft)`. Runs prompt in one forward pass. Returns last-position logits and wall time for TTFT (time to first token) observability.
- **`engine/decode.py`** — `decode_step(model, last_token_id, kv_cache, sampler, **sampler_kwargs) → next_token_id`. Single autoregressive step. The caller provides the decoded token from the previous step.
- **`engine/generate.py`** — `GenerateEngine(model, tokenizer, sampler, eos_token_id)`. Class that holds config, manages KV cache lifecycle, and exposes `async generate(prompt, max_tokens, **sampler_kwargs) → AsyncIterator[str]`.

## Consequences

- Each pipeline step is independently testable, mockable, and replaceable
- Prefill reuse across generations is natural — callers can pass a cached `kv_cache` in
- Class-based engine vs stateless functions: the class owns cache lifecycle and logging state; the functions remain pure operations over their inputs
- KV cache is owned by `GenerateEngine` internally, with a `reset()` method for reuse. This avoids forcing cache lifecycle on callers.
- Token-by-token streaming gives callers maximum control (useful for UI debouncing, stop-word detection, logging). Chunking can be layered on top.
- Async generator (`AsyncIterator[str]`) supports concurrent request handling in future serving layers

## References

- ADR 001: Registry Pattern (tok/sampler dispatch)
- ADR 003: PyTorch-Only Backend (no extra async runtime needed)
- vLLM AsyncEngine — inspiration for the async generate interface
