# snapmind — Framework Agent Config

## How to Add a New Component

1. Create an ABC in the appropriate `base.py` file (or add to existing one)
2. Create the implementation file with `@REGISTRY.register("key")` decorator
3. Import the implementation in `snapmind/__init__.py` to trigger registration
4. Add tests: ABC contract + mathematical property + architecture conformance
5. If adding new layer types, add them to `layers/<type>/<name>.py`

## Registry Reference

| Component | Registry | ABC Location | Key Convention |
|---|---|---|---|
| Attention | `ATTENTION` | `layers/attention/base.py` | `"sdpa"`, `"gqa"` |
| Positional Encoding | `PE` | `layers/positional/base.py` | `"learned"`, `"rope"`, `"none"` |
| Normalization | `NORM` | `layers/normalization/base.py` | `"layernorm"`, `"rmsnorm"` |
| Activation | `ACTIVATION` | `layers/activation/base.py` | `"gelu"`, `"silu"` |
| KV Cache | `KV_CACHE` | `kv_cache/base.py` | `"naive"` |
| Tokenizer | `TOKENIZER` | `tokenizer/base.py` | `"hf"` |
| Sampler | `SAMPLER` | `sampling/base.py` | `"greedy"`, `"top_p"`, `"temperature"` |
| Weight Loader | `LOADER` | `loaders/base.py` | `"safetensors"` |
| Model | `MODEL` | `models/base.py` | `"gpt2"`, `"llama"` |

## Layer Hierarchy

```
TransformerBlock
├── Norm (input)
├── Attention (pluggable via config.attention_type)
│   ├── Q, K, V projections
│   ├── PE.apply_to_qk (for RoPE)
│   └── Scaled dot-product / GQA
├── Residual +
├── Norm (post-attention)
├── FeedForward / GatedFeedForward (pluggable via model choice)
└── Residual +
```

## Test Pattern

```python
# Category 1: ABC Contract
def test_cannot_instantiate_directly():
    with pytest.raises(TypeError):
        SomeABC()

# Category 2: Mathematical Property
def test_output_property():
    result = component(input)
    assert some_invariant(result)

# Category 3: Architecture Conformance
def test_model_property():
    assert model.lm_head.weight is model.embed.weight  # GPT-2 weight tying
```
