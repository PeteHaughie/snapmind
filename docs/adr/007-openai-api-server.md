# ADR 007: Registry-Native OpenAI API Server

## Status

Proposed

## Context

snapmind has an OpenAI-compatible API server (`serving/openai_api.py`) that exposes `/v1/models`, `/v1/chat/completions`, and streaming SSE output. The current implementation works but breaks the framework's core plugin principle by importing concrete implementations directly:

```python
# Current code — hardcoded imports:
from snapmind.sampling.greedy import GreedySampler
from snapmind.sampling.top_p import TopPSampler
from snapmind.tokenizer.hf import HFTokenizer
```

This means:
- Adding a new sampler (Top-K, Mirostat) requires editing the API server code
- The API server has privileged knowledge of component internals
- User-registered plugins are invisible to the API

The OpenAI `/v1/chat/completions` schema defines sampling via `temperature` and `top_p` parameters but has no standard fields for `top_k`, `mirostat`, or other strategies. We need to decide how to map OpenAI API params to our registry dispatch without inventing a contradictory schema.

## Decision Drivers

- The API server must dispatch through `SAMPLER` and `TOKENIZER` registries
- The REST schema should not diverge unnecessarily from the OpenAI standard
- Users who register custom samplers should be able to use them via the API
- Streaming and non-streaming responses must behave identically
- The server should not import concrete classes from any layer

## Considered Approaches

### Option 1: Pure OpenAI-mapped dispatch

Map OpenAI params to sampler registry keys:

| OpenAI params | SAMPLER.create() call |
|---|---|
| Neither temp nor top_p | `"greedy"` |
| `temperature` only | `"temperature"` with `temperature=req.temperature` |
| `top_p` ± `temperature` | `"top_p"` with `top_p=req.top_p, temperature=req.temperature` |
| `top_p` + `top_k` | Already invalid per OpenAI spec (no top_k field) |

Tokenization via `TOKENIZER.create("hf", model_name=...)`. Extensions (top_k, mirostat) are not exposed on the REST schema — users who need them use the Python API or CLI directly.

### Option 2: Extended schema with aliased params

Extend the ChatRequest body with OpenRouter-style generation params:

```python
class ChatRequest(BaseModel):
    ...
    top_k: int | None = None  # additional param, not in OpenAI spec
    mirostat_tau: float | None = None
    mirostat_learning_rate: float | None = None
```

Server-side dispatch:

| Params set | SAMPLER key |
|---|---|
| `mirostat_tau` set | `"mirostat"` with `tau, learning_rate` |
| `top_k` set (no top_p) | `"top_k"` with `top_k` |
| `top_k` + `top_p` | `"top_k"` (top_p ignored, or compose — TBD) |
| Only `temperature` | `"temperature"` |
| Only `top_p` | `"top_p"` |
| Neither | `"greedy"` |

The schema is still recognizable to OpenAI clients (all standard fields work identically) but accepts extra fields for snapmind-native features.

### Option 3: Explicit `sampler` field

Add an explicit `sampler: str = "auto"` field that overrides automatic dispatch:

```json
{
  "model": "llama",
  "messages": [...],
  "sampler": "mirostat",
  "mirostat_tau": 2.0,
  "mirostat_learning_rate": 0.1
}
```

When `sampler="auto"`, dispatch from OpenAI params (Option 1 logic). When explicitly set, use the named registry key directly and pass all matching params.

## Decision

Adopt **Option 2 (Extended schema)** with the following rules:

1. All standard OpenAI fields (`temperature`, `top_p`, `max_tokens`, `stream`) work identically to the spec
2. Additional fields (`top_k`, `mirostat_tau`, `mirostat_learning_rate`) are accepted but absent from OpenAPI schema generation to avoid confusing generic clients
3. Sampler selection priority (first match wins):
   - `mirostat_tau` set → `SAMPLER.create("mirostat", tau=mirostat_tau, learning_rate=mirostat_learning_rate or 0.1)`
   - `top_k` set (no top_p) → `SAMPLER.create("top_k", k=top_k)`
   - `top_p` set → `SAMPLER.create("top_p", top_p=top_p, temperature=temperature or 1.0)`
   - Only `temperature` (≠ 1.0) → `SAMPLER.create("temperature", temperature=temperature)`
   - None of the above → `SAMPLER.create("greedy")`
4. Tokenization always uses `TOKENIZER.create("hf", model_name=...)`

## Consequences

### Positive

- All component dispatch goes through registries — any registered sampler is reachable via API
- Standard OpenAI clients work unchanged; extended fields are invisible to them
- No concrete imports in the API server
- Adding a new sampler type is zero-cost for the API (only needed if you want REST-level params)

### Negative

- The REST schema is technically non-standard (extra fields accepted even if undocumented)
- Sampler priority rules must be documented explicitly
- Some parameter combinations are silently ignored (e.g., `top_k` + `top_p`, only `top_k` takes effect)
- Mirostat's `tau` and `learning_rate` don't map to any OpenAI concept — callers must know snapmind's API

### Risks

- Generic OpenAI clients that validate strictly may reject extra fields (mitigation: use Pydantic's `extra="ignore"`)
- Mirostat is stateful per-session — the server must manage sampler instances, not just stateless constructor calls. Mitigation: maintain a lightweight sampler registry keyed by request ID.

## Implementation Notes

- Use `model_config = {"extra": "ignore"}` on ChatRequest to silently drop unknown fields
- The `/v1/chat/completions` endpoint should accept `GenerateEngine` or engine config as a dependency, not construct it inline
- Sampler dispatch: implement a `resolve_sampler(req: ChatRequest)` function in `serving/` that returns `(sampler, sampler_kwargs)` without importing concrete classes
- Mirostat session state: attach sampler instance to the streaming SSE generator's scope; for non-streaming, allocate and discard per request

## Unresolved Questions

- [ ] Should `top_k` and `top_p` compose (apply both filters) or be mutually exclusive? Current rule: top_k alone → top_k, top_p alone → top_p, both → only top_k takes effect.
- [ ] Should the extended schema be advertised in the `/v1/models` response (via `extra_body` or similar)?
- [ ] EOS token (`tokenizer.eos_token_id`) — should it be configurable per request or model-defined?

## References

- [OpenAI Chat Completion API Reference](https://platform.openai.com/docs/api-reference/chat)
- [OpenRouter Generation Parameters](https://openrouter.ai/docs/parameters) — precedent for extended OpenAI schema
- ADR 001: Registry Pattern (dispatcher architecture)
- ADR 005: Engine Pipeline Design (GenerateEngine composition)
- Current file: `serving/openai_api.py` (to be rewritten)
