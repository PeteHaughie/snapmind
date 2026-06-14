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
| `layers/positional/base.py` | `PositionalEncodingABC` | Abstract contract with `injection_point` |
| `layers/positional/learned.py` | `LearnedPositionalEncoding` | GPT-2 style absolute learned PE |
| `layers/positional/none.py` | `NoPositionalEncoding` | Pass-through |
| `layers/normalization/base.py` | `NormABC` | Abstract contract |
| `layers/normalization/layer_norm.py` | `LayerNorm` | Used by GPT-2, BERT |
| `layers/activation/base.py` | `ActivationABC` | Abstract contract |
| `layers/activation/gelu.py` | `GELU` | Used by GPT-2 |
| `layers/feed_forward.py` | `FeedForward` | FFN using pluggable activation |
| `layers/transformer_block.py` | `TransformerBlock` | Composes norm → attn → norm → ffn using config |
| `models/base.py` | `BaseModelABC` | Abstract model contract |
| `models/gpt2.py` | `GPT2Model` | GPT-2 124M architecture: learned PE, LayerNorm, GELU, MHA, weight tying |

**Test files:**

| File | Coverage |
|---|---|
| `tests/conftest.py` | Shared fixtures: `tiny_config`, `tiny_gpt2`, `test_tokens` |
| `tests/test_registry.py` | Registry: register, create, list, error cases, decorator |
| `tests/test_config.py` | Config: defaults, field validation, serialization round-trip |
| `tests/layers/attention/test_base.py` | ABC contract: cannot instantiate, minimal subclass |
| `tests/layers/attention/test_sdpa.py` | SDPA: attention weights sum to 1, causal mask isolates future |
| `tests/layers/positional/test_base.py` | ABC contract |
| `tests/layers/positional/test_learned.py` | Learned PE: different positions → different vectors |
| `tests/layers/normalization/test_base.py` | ABC contract |
| `tests/layers/normalization/test_layer_norm.py` | LayerNorm: standardized output, affine shift |
| `tests/layers/activation/test_base.py` | ABC contract |
| `tests/layers/activation/test_gelu.py` | GELU: correct values at -1, 0, 1 |
| `tests/layers/test_feed_forward.py` | FFN: output shape, activation applied |
| `tests/layers/test_transformer_block.py` | Block: shape conservation, residual preserved |
| `tests/models/test_base.py` | Model ABC contract |
| `tests/models/test_gpt2.py` | GPT-2: param count, weight tying, causal mask, logit shape |

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

**Test files:**

| File | Coverage |
|---|---|
| `tests/kv_cache/test_base.py` | KVCacheABC contract |
| `tests/kv_cache/test_naive.py` | Naive: store/fetch round-trip, append grows sequence |
| `tests/tokenizer/test_base.py` | TokenizerABC contract |
| `tests/tokenizer/test_hf.py` | HF: encode/decode round-trip, known token IDs |
| `tests/sampling/test_base.py` | SamplerABC contract |
| `tests/sampling/test_greedy.py` | Greedy: argmax property |
| `tests/sampling/test_top_p.py` | Top-p: cumulative probability threshold |
| `tests/loaders/test_base.py` | WeightLoaderABC contract |
| `tests/loaders/test_safetensors.py` | Safetensors: round-trip save/load |
| `tests/engine/test_prefill.py` | Prefill: parallel processing correct |
| `tests/engine/test_generate.py` | Generate: full pipeline produces tokens |

**Integration:** Load GPT-2 124M weights → generate text → compare against reference output.

---

### Phase 3: Broader Coverage

**Goal:** Support multiple model families and advanced KV cache strategies.

| Module | Files | Deliverable |
|---|---|---|
| `models/llama.py` | `LlamaModel` | Pre-norm, RoPE, SwiGLU, GQA, RMSNorm |
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
| Tests | Property-based tests, regression tests, end-to-end pipeline tests | All components tested in isolation + integration across all 4 categories |

---

## Testing Strategy

Four categories of tests, each proving something different about the framework:

### 1. ABC Contract Tests

Prove the base class design is sound — that every ABC correctly enforces its contract:

```python
def test_cannot_instantiate_directly():
    with pytest.raises(TypeError):
        AttentionABC()

def test_minimal_subclass_works():
    class MyAttn(AttentionABC):
        def forward(self, x, kv_cache, pos_ids, mask): return x
    assert MyAttn().forward(t, None, None, None).shape == (1, 4, 64)
```

### 2. Mathematical Property Tests

Prove implementations match their mathematical specification — these tests hold regardless of the model using the component:

```python
def test_attention_weights_sum_to_one():
    output = attn(x)
    assert torch.allclose(weights.sum(dim=-1), torch.ones_like(weights))

def test_causal_mask_prevents_future():
    output = attn(x, mask=causal_mask)
    assert (attn.last_weights.triu(diagonal=1) == 0).all()

def test_layernorm_standardizes():
    out = layer_norm(tensor)
    assert torch.allclose(out.mean(), 0.0, atol=1e-5)
    assert torch.allclose(out.std(), 1.0, atol=1e-5)
```

### 3. Architecture Conformance Tests

Prove a model implementation matches its reference architecture's known properties:

```python
def test_gpt2_parameter_count():
    model = GPT2Model(config)
    n = sum(p.numel() for p in model.parameters())
    assert abs(n - 124_000_000) < 1_000_000

def test_weight_tying():
    assert model.lm_head.weight is model.embed.weight
```

### 4. Principle-Based Algorithm Tests

Prove algorithms maintain their invariants — KV cache strategies, quantization, sampling:

```python
def test_store_fetch_round_trip():
    cache.store(0, k, v, 0)
    k_out, v_out = cache.fetch(0)
    assert torch.equal(k, k_out) and torch.equal(v, v_out)

def test_top_p_cumulative_threshold():
    probs = F.softmax(logits, dim=-1)
    assert probs[sampled_idx] >= 1.0 - top_p  # no improbable token selected
```

### Test Directory Structure

```
snapmind/tests/
├── conftest.py                     # Shared fixtures
├── test_registry.py                # Registry unit tests
├── test_config.py                  # Config unit tests
├── layers/
│   ├── attention/
│   │   ├── test_base.py
│   │   ├── test_sdpa.py
│   │   └── test_gqa.py
│   ├── positional/
│   │   ├── test_base.py
│   │   ├── test_rope.py
│   │   └── test_learned.py
│   ├── normalization/
│   │   ├── test_base.py
│   │   ├── test_rms_norm.py
│   │   └── test_layer_norm.py
│   ├── activation/
│   │   ├── test_base.py
│   │   └── test_gelu.py
│   ├── test_feed_forward.py
│   └── test_transformer_block.py
├── models/
│   ├── test_base.py
│   ├── test_gpt2.py
│   └── test_llama.py
├── kv_cache/
│   ├── test_base.py
│   ├── test_naive.py
│   └── test_paged.py
├── tokenizer/
│   ├── test_base.py
│   └── test_hf.py
├── sampling/
│   ├── test_base.py
│   ├── test_greedy.py
│   ├── test_top_k.py
│   └── test_top_p.py
├── loaders/
│   ├── test_base.py
│   ├── test_safetensors.py
│   └── test_pytorch.py
└── engine/
    ├── test_prefill.py
    └── test_generate.py
```

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
