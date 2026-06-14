# snapmind — Implementation Plan

## Scope

A pure-Python, modular transformer inference framework. No compiled extensions. Every component is a hot-swappable plugin registered by string key.

## Phases

### Phase 0: Project Scaffolding

**Goal:** Establish conventions, agent configuration, ADRs, and experiment tracking before writing code.

| Module | Files | Deliverable |
|---|---|---|
| Agent config | `.agents/AGENTS.md` | Project-level agent instructions |
| Framework agent config | `snapmind/.agents/AGENTS.md`, `SYSTEM.md`, `MEMORY.md` | Framework-specific agent config |
| ADRs | `docs/adr/README.md`, `001`-`004` | 4 initial ADRs covering core architectural decisions |
| Experiment tracking | `docs/experiments/README.md` | Convention for one-folder-per-paper experiment logging |
| Session memory | `MEMORY.md` (root) | Persistent agent context between sessions |
| Source annotations | Applied across all source files in later phases | SECTION/ENDSECTION and ANCHOR/ENDANCHOR markers |

---

### Phase 1: Foundation + Layers

**Goal:** The framework can instantiate a model with any combination of pluggable components.

| Module | Files | Deliverable |
|---|---|---|
| `core/registry.py` | `Registry[T]` generic class | Registration via decorator and direct call; `create()` dispatch; `list()` introspection |
| `core/config.py` | `ModelConfig`, `EngineConfig`, `KVCacheConfig` dataclasses | Single config object flows through the entire system |
| `layers/attention/base.py` | `AttentionABC` | Abstract contract for all attention variants |
| `layers/attention/sdpa.py` | `ScaledDotProductAttention` | Default MHA implementation |
| `layers/attention/gqa.py` | `GroupedQueryAttention` | GQA variant (used by Llama 3, Mistral) |
| `layers/positional/base.py` | `PositionalEncodingABC` | Abstract contract with `injection_point` |
| `layers/positional/rope.py` | `RotaryPositionalEncoding` | RoPE (used by Llama, Mistral, GPT-NeoX) |
| `layers/positional/none.py` | `NoPositionalEncoding` | Pass-through |
| `layers/normalization/base.py` | `NormABC` | Abstract contract |
| `layers/normalization/rms_norm.py` | `RMSNorm` | Used by Llama, Mistral |
| `layers/normalization/layer_norm.py` | `LayerNorm` | Used by GPT-2, BERT |
| `layers/activation/base.py` | `ActivationABC` | Abstract contract |
| `layers/activation/swiglu.py` | `SwiGLU` | Gated activation used by Llama |
| `layers/activation/gelu.py` | `GELU` | Used by GPT-2 |
| `layers/feed_forward.py` | `FeedForward`, `GatedFFN` | FFN using pluggable activation |
| `layers/transformer_block.py` | `TransformerBlock` | Composes norm → attn → norm → ffn using config |
| `models/base.py` | `BaseModelABC` | Abstract model contract |
| `models/llama.py` | `LlamaModel` | Full Llama architecture (pre-norm, RoPE, SwiGLU, GQA) |

**Tests:** Unit tests for each layer + registry + config instantiation.

---

### Phase 2: Weights + Inference

**Goal:** The framework can load real model weights and generate tokens.

| Module | Files | Deliverable |
|---|---|---|
| `kv_cache/base.py` | `KVCacheABC` | Abstract cache contract |
| `kv_cache/naive.py` | `NaiveKVCache` | Simple concatenation per layer |
| `tokenizer/base.py` | `TokenizerABC` | Abstract tokenizer contract |
| `tokenizer/hf.py` | `HuggingFaceTokenizer` | Wraps `tokenizers` library |
| `sampling/base.py` | `SamplerABC` | Abstract sampler contract |
| `sampling/greedy.py` | `GreedySampler` | argmax sampling |
| `sampling/top_p.py` | `TopPSampler` | Nucleus sampling |
| `sampling/temperature.py` | `TemperatureSampler` | Temperature-scaled sampling |
| `loaders/base.py` | `WeightLoaderABC` | Abstract loader contract |
| `loaders/safetensors.py` | `SafetensorsLoader` | Loads `.safetensors` files |
| `loaders/pytorch.py` | `PyTorchLoader` | Loads `.bin` state dicts |
| `engine/prefill.py` | `prefill()` | Parallel prompt processing |
| `engine/decode.py` | `decode_step()` | Single autoregressive step |
| `engine/generate.py` | `generate()` | Async generator, streaming output |
| `core/logging.py` | `InferenceLogger` | Token/s, TTFT, per-token timing |

**Tests:** Integration test loading a small model (e.g., GPT-2 124M), verifying output matches expected tokens. Benchmark baseline.

---

### Phase 3: Broader Coverage

**Goal:** Support multiple model families and advanced KV cache strategies.

| Module | Files | Deliverable |
|---|---|---|
| `models/gpt2.py` | `GPT2Model` | Post-norm, learned PE, GELU, MHA |
| `models/mistral.py` | `MistralModel` | Pre-norm, RoPE, SwiGLU, sliding window |
| `kv_cache/paged.py` | `PagedKVCache` | Block-based allocation, prefix caching |
| `kv_cache/sliding_window.py` | `SlidingWindowKVCache` | Used by Mistral |
| `sampling/top_k.py` | `TopKSampler` | Top-k sampling |
| `sampling/mirostat.py` | `MirostatSampler` | Adaptive sampling |
| `layers/attention/mla.py` | `MultiHeadLatentAttention` | DeepSeek-style MLA |

---

### Phase 4: Serving + Polish

**Goal:** Production-ready CLI and OpenAI-compatible server.

| Module | Files | Deliverable |
|---|---|---|
| `serving/cli.py` | CLI with `serve`, `generate`, `list` commands | `python -m snapmind serve --model llama --port 8000` |
| `serving/openai_api.py` | FastAPI server | `/v1/chat/completions`, streaming, token usage |
| Documentation | In-code docstrings, README usage examples | Every ABC, registry, and config documented |
| Tests | Property-based tests, regression tests | All components tested in isolation + integration |

---

## Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Backend | PyTorch only | Abstracting too early adds complexity without benefit |
| Architecture style | Per-file models | Explicit, easy to debug, follows HF Transformers convention |
| Registry scope | Global singletons per type | Simple, familiar, proven by FlexiTransformers and HF |
| Config | Single `ModelConfig` dataclass | One object flows through entire system; never wonder what config applies where |
| Async | Async-native engine | Streaming generation is inherently async; continuous batching benefits |
| Dependencies | Minimal: PyTorch + `tokenizers` | Quantization and serving extras are `[quant]`, `[serve]` optional extras |
| AI navigation | SECTION/ANCHOR markers, `.agents/`, ADRs, `docs/experiments/` | Most engineering will be done by AI agents; codebase must be parseable programmatically |

## AI-Friendly Design

snapmind is built for an era where most engineering is performed by AI coding agents. Every convention serves agent navigation:

- **SECTION/ENDSECTION markers** let agents grep for the right code block without reading entire files
- **ANCHOR/ENDANCHOR markers** mark key decision points (class boundaries, plugin hooks, config boundaries)
- **`docs/adr/`** prevents agents from making contradictory decisions across sessions
- **`docs/experiments/`** creates a searchable paper trail: what was tried, what worked, what didn't, and why
- **`MEMORY.md`** persists session context so agents don't lose focus between invocations
- **`.agents/` directories** hold tool-specific instructions at both project and framework level

## Inspirations

- **FlexiTransformers** — pluggable PE registry via `register_pe()` and the ABC+`injection_point` pattern for dispatching at the right moment in attention.
- **HuggingFace Transformers** — per-file model definitions, `register_model` pattern, weight name mapping conventions.
- **Karpathy's nanoGPT / llama2.py** — philosophy that inference should be understandable in a single sitting.
- **vLLM** — PagedAttention concept, reimplemented as a pluggable Python component rather than a compiled C++ kernel.
- **Comment Anchors (VSCode extension)** — SECTION/ANCHOR markers inspired by navigable code annotations.
- **CLAUDE.md conventions** — Memory files and agent configuration for session continuity.
- **MADR** — Markdown Architecture Decision Records for structured decision logging.
