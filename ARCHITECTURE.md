# snapmind Architecture

## Overview

snapmind is a modular pure-Python framework for local transformer inference. Every component implements an abstract base class and registers itself by string key. The framework never imports concrete implementations — it dispatches through registries. Users swap components by changing a single field in `ModelConfig`.

```
Framework Code          User Code
┌──────────────────┐    ┌──────────────────────────┐
│  Registry        │    │  register_kv_cache(      │
│  dispatches by   │◄───│    "sentence_eviction",  │
│  string key      │    │    SentenceEvictionCache) │
└────────┬─────────┘    └──────────────────────────┘
         │
┌────────▼──────────────────────────────────────────┐
│  Inference Pipeline (registry-agnostic)           │
│  Calls attention via ABC, never knows the         │
│  concrete implementation                          │
└───────────────────────────────────────────────────┘
```

## Core Principle

**Everything is a plugin.** The framework is a composition of independently swappable blocks held together by registries. There is no monolithic "inference engine" — there is a pipeline that asks registries for components by name.

## Package Structure

```
snapmind/
├── .agents/                    # Agent config (instructions, system prompt, memory)
├── core/                       # Foundation: registries, configs
├── models/                     # Full model architectures (composition of layers)
├── layers/                     # Primitive building blocks
│   ├── attention/              # Pluggable attention mechanisms
│   ├── positional/             # Pluggable positional encodings
│   ├── normalization/          # Pluggable norm layers
│   └── activation/             # Pluggable activation functions
├── kv_cache/                   # Pluggable KV cache strategies
├── tokenizer/                  # Pluggable tokenization
├── sampling/                   # Pluggable token selection strategies
├── engine/                     # Prefill, decode, generate pipeline
├── loaders/                    # Weight format loaders
├── serving/                    # CLI and HTTP server
├── docs/
│   ├── adr/                    # Architecture Decision Records
│   └── experiments/            # One folder per tested paper/idea
├── MEMORY.md                   # Session-persistent agent context (root)
└── .agents/AGENTS.md           # Project-level agent instructions (root)
```

## Component Ownership

Every swappable component type has an ABC, a Registry singleton, and a default implementation:

| Component | ABC | Registry | Default | Injection Point |
|---|---|---|---|---|
| Attention | `AttentionABC` | `ATTENTION` | `"sdpa"` | Within `TransformerBlock` |
| Positional encoding | `PositionalEncodingABC` | `PE` | `"rope"` | Inside attention (Q/K or scores) |
| Normalization | `NormABC` | `NORM` | `"rmsnorm"` | Before/after sublayers in `TransformerBlock` |
| Activation | `ActivationABC` | `ACTIVATION` | `"swiglu"` | Inside `FeedForward` / `GatedFFN` |
| KV cache | `KVCacheABC` | `KV_CACHE` | `"naive"` | Inside attention, per-layer |
| Sampler | `SamplerABC` | `SAMPLER` | `"greedy"` | After `model.forward()` returns logits |
| Tokenizer | `TokenizerABC` | `TOKENIZER` | `"hf"` | At pipeline boundaries (encode/decode) |
| Weight loader | `WeightLoaderABC` | `LOADER` | `"safetensors"` | During `load_model()` |
| Model | `BaseModelABC` | `MODEL` | (none) | Entry point for forward pass |

## The Registry Pattern

The `Registry[T]` class is the linchpin of the entire plugin system. One instance per component type.

```python
class Registry[T]:
    """Generic plugin registry. Type-safe dispatch by string key."""

    def __init__(self, name: str): ...
    def register(self, key: str, cls: type[T], *, override: bool = False): ...
    def create(self, key: str, **kwargs) -> T: ...
    def list(self) -> list[str]: ...
```

**Registration** can be either decorator or direct:

```python
# Decorator (used in implementation files):
@ATTENTION.register("gqa")
class GroupedQueryAttention(AttentionABC): ...

# Direct (used by users adding custom components):
ATTENTION.register("my_custom", MyCustomAttention)
```

**Dispatch** happens in `create()`:

```python
# Framework code — never imports concrete classes:
attn = ATTENTION.create(
    config.attention_type,
    d_model=config.d_model,
    n_heads=config.n_heads,
    n_kv_heads=config.n_kv_heads or config.n_heads,
)
```

## Configuration

A single `ModelConfig` dataclass flows through the entire system. Component choice is driven by string keys:

```python
@dataclass
class ModelConfig:
    model_type: str = "llama"         # Key into MODEL registry
    d_model: int = 4096
    n_heads: int = 32
    n_kv_heads: int | None = None     # None = same as n_heads (MHA)
    n_layers: int = 32
    d_ff: int | None = None           # None = inferred as 4*d_model
    vocab_size: int = 128256
    max_seq_len: int = 8192
    norm_eps: float = 1e-5

    # Pluggable component selection:
    attention_type: str = "sdpa"
    pe_type: str = "rope"
    norm_type: str = "rmsnorm"
    activation_type: str = "swiglu"
    kv_cache_type: str = "naive"
```

## Data Flow: End-to-End Generation

```
User calls:
  generate(model, tokenizer, "What is attention?")

1. tokenizer.encode("What is attention?")
   → [7453, 427, 8912]              # TokenizerABC.encode()

2. model.forward(tokens, kv_cache, position_ids)
   ↓
   for each layer in model.layers:  # nn.ModuleList of TransformerBlock
   ↓
   a) norm1(x)                      # NORM.create(config.norm_type)
   b) attention(norm1(x),           # ATTENTION.create(config.attention_type)
         kv_cache=kv_cache[layer_i],
         position_ids=...)
        ↓
        - Project Q, K, V
        - Apply PE to Q/K           # PE.create(config.pe_type).apply_to_qk()
        - Store K,V in cache        # kv_cache[layer_i].store(k, v, pos)
        - Compute scores
        - Apply PE to scores        # PE.apply_to_scores()
        - softmax, weighted sum
        - Output projection
        ↓
   c) x = x + attn_out             # Residual
   d) norm2(x)                     # NORM.create(config.norm_type)
   e) ffn(norm2(x))                # uses ACTIVATION.create(config.activation_type)
        ↓
        - Gate projection           # if GatedFFN
        - Activation                # ACTIVATION.create().forward()
        - Down projection
        ↓
   f) x = x + ffn_out             # Residual
   ↓
   h = model.norm(x)               # Final norm
   logits = model.lm_head(h)       # Linear projection to vocab

3. sampler.sample(logits[:, -1])   # SAMPLER.create(config.sampler_type)
   → next_token_id

4. tokenizer.decode([next_token])  # TokenizerABC.decode()
   → "Attention"                   # Yielded to caller

5. Repeat from step 2 with kv_cache
```

## Points of Pluggability

In a single forward pass, these components are independently swappable:

| Step | Component | Registry Key |
|---|---|---|
| Tokenization | `tokenizer.encode()` / `decode()` | `TOKENIZER` |
| Positional encoding on Q/K | `pe.apply_to_qk()` | `PE` |
| KV cache store/fetch | `kv_cache.store()` / `fetch()` | `KV_CACHE` |
| Attention computation | `attention.forward()` | `ATTENTION` |
| Positional encoding on scores | `pe.apply_to_scores()` | `PE` |
| Normalization | `norm.forward()` | `NORM` |
| Activation function | `activation.forward()` | `ACTIVATION` |
| Token selection | `sampler.sample()` | `SAMPLER` |

## Weight Loading Architecture

Weight loaders map weight names from file formats to model parameter names:

```python
@register("loader", "safetensors")
class SafetensorsLoader(WeightLoaderABC):
    def load(self, path: str, model: BaseModelABC, config: ModelConfig):
        # 1. Load .safetensors file
        # 2. Get weight map for this model type
        #    e.g., llama: "model.layers.0.input_layernorm.weight" → "layers.0.norm1.weight"
        # 3. Assign to model parameters
```

The weight map is the most architecture-specific part — each model family stores weights under different names. Maps live alongside model definitions.

## Async Design

The engine is async-native from the ground up:

```python
async def generate(...) -> AsyncIterator[str]:
    # prefill (one call)
    # decode loop (yield tokens as they're generated)
```

This enables:
- Streaming output token-by-token
- Concurrent requests (future: continuous batching)
- Cancellation

## Inspirations for Architectural Decisions

### FlexiTransformers — Registry + Injection Point Pattern

FlexiTransformers' `register_pe()` system and the `injection_point` property on `PositionalEncoding` is the direct inspiration for snapmind's entire plugin architecture. FlexiTransformers proved that an ABC with a well-defined injection point (embedding, Q/K, or scores) lets attention code stay clean while supporting radically different PE variants. snapmind generalizes this: every component type has a registry and an ABC with a clear contract.

### HuggingFace Transformers — Per-File Model Definitions

HuggingFace's convention of one Python file per architecture (modeling_llama.py, modeling_gpt2.py, etc.) keeps each model family self-contained and easy to find. snapmind adopts the same structure in `models/`.

### nanoGPT / llama2.py (Karpathy) — Simplicity as a Feature

Karpathy's implementations demonstrate that transformer inference is fundamentally simple — it's the optimization frameworks that make it complex. snapmind prioritizes readable, minimal core code (~5K lines for the framework) so that a developer can understand the full stack in an hour.

### vLLM PagedAttention — Modular Reimplementation

vLLM's PagedAttention is the standard for KV cache memory management, but it's compiled into C++/CUDA kernels. snapmind's `PagedKVCache` reimplements the same idea in pure Python as a swappable plugin — demonstrating that even "kernel-level" optimizations can be pluggable components.

## AI-Friendly Design

snapmind is built for an era where most engineering is performed by AI coding agents. Every convention exists to make the codebase navigable programmatically.

### Source Annotations

Source files use two marker types for AI navigation:

```python
# ─── SECTION: Attention Base ───────────────────────────
# ANCHOR: class AttentionABC
class AttentionABC(ABC):
    """Every attention variant implements this."""
    @abstractmethod
    def forward(self, x, kv_cache, position_ids, mask): ...
# ENDANCHOR: class AttentionABC
# ─── ENDSECTION: Attention Base ────────────────────────

# ─── SECTION: Scaled Dot-Product Attention ─────────────
# ANCHOR: class SDAAttention
class ScaledDotProductAttention(AttentionABC):
    ...
# ENDANCHOR: class SDAAttention
# ─── ENDSECTION: Scaled Dot-Product Attention ──────────
```

- `SECTION` / `ENDSECTION` — large block boundaries (entire module, class, or logical section)
- `ANCHOR` / `ENDANCHOR` — key decision points inside sections (class definitions, plugin hooks, config boundaries)

### Agent Configuration

| File | Purpose |
|---|---|
| `.agents/AGENTS.md` (root) | Project-level agent instructions |
| `snapmind/.agents/AGENTS.md` | Framework-specific instructions for contributors |
| `snapmind/.agents/SYSTEM.md` | System prompt additions for LLM tooling |
| `snapmind/.agents/MEMORY.md` | Session-persistent agent context (current focus, open questions) |
| `MEMORY.md` (root) | Top-level session memory |

### Architecture Decision Records

Every significant decision is captured in `docs/adr/` as a MADR-format document:

```
docs/adr/
├── README.md              # Index of all ADRs
├── 001-use-registry-pattern.md
├── 002-per-file-model-architectures.md
├── 003-pytorch-only-backend.md
└── 004-ai-friendly-annotations.md
```

ADRs prevent agents from making contradictory decisions across sessions and provide a searchable rationale for every choice.

### Experiment Tracking

Every paper-based implementation is logged in `docs/experiments/<name>/`:

```
docs/experiments/sentence-level-kv-eviction/
├── README.md      # Paper ref (arXiv/HF link), status, results, rejection reason
├── notes.md       # Freeform implementation diary
└── results.tsv    # experiment / metric_before / metric_after / status
```

Status values: `accepted` (merged), `experimental` (implemented but not merged), `rejected` (tested and discarded), `pending` (planned).

## Dependencies

Core (required):
- `torch` >= 2.0
- `tokenizers` (HuggingFace)

Optional:
- `safetensors` (weight loading)
- `fastapi` + `uvicorn` (serving)
- `prometheus-client` (metrics)
