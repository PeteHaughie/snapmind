# ADR 006: Llama Model Implementation

## Status

Accepted

## Context

snapmind's first model (GPT-2) validates the registry pattern and core layers. To prove the framework generalizes to other architectures, we need a second, structurally different model family. Llama is the most popular open-weight family and differs from GPT-2 in every pluggable dimension: positional encoding (RoPE vs Learned PE), normalization (RMSNorm vs LayerNorm), activation (SiLU in gated FFN vs GELU in standard FFN), attention (GQA vs MHA), and weight conventions (no bias, no weight tying).

## Decisions

### 1. Separate LlamaTransformerBlock

Instead of loading TransformerBlock with config flags (`ffn_gated`, `use_bias`, etc.), LlamaModel defines its own `LlamaTransformerBlock` inside `models/llama.py`. This keeps each architecture's block self-contained and independently auditable. The common residual pattern (pre-norm → sublayer → residual) is duplicated, but the blocks differ in enough details (FFN structure, bias conventions, attention wiring) that a shared base would require substantial abstraction.

### 2. `apply_to_qk()` on PositionalEncodingABC

RoPE is applied to Q and K inside the attention layer, not to the embedding. We add `apply_to_qk(q, k, position_ids)` to the ABC with a default no-op implementation (`return q, k`). Existing LearnedPE and NoPositionalEncoding subclasses work unchanged. This extension is backward compatible — no existing code needs modification.

### 3. RoPE injected into GQA via constructor

`GroupedQueryAttention` accepts an optional `pe` parameter. After projecting Q and K (and splitting into heads), it calls `pe.apply_to_qk(q, k, position_ids)`. This keeps RoPE logic in the PE class and attention logic in the attention class. Future attention variants can support RoPE by accepting the same optional `pe` parameter.

### 4. GatedFeedForward as a separate layer

Llama's gated FFN (`down_proj(silu(gate_proj(x)) * up_proj(x))`) uses three linear projections versus GPT-2's two. A separate `GatedFeedForward` class avoids conditional logic (`if gated:`) in a shared class. Both can coexist and be reused by future models (e.g., Mixtral uses gated FFN, BERT uses standard FFN).

### 5. No weight tying, no bias for Llama

Llama does not tie `lm_head` weights to the embedding table and uses `bias=False` on all linear projections. These defaults differ from GPT-2 and are set explicitly in the model definition.

### 6. SiLU registered as a separate activation

SiLU (`x * sigmoid(x)`) is the activation function inside Llama's gated FFN. It is registered as `"silu"` in the ACTIVATION registry. It is structurally simple and reusable by any model that uses SiLU/Swish.

## Consequences

- Positive: Each model family is self-documenting; no config flag sprawl in shared classes.
- Positive: RoPE can be reused by any attention variant that accepts an optional PE object.
- Positive: GatedFeedForward can be reused by future gated-FFN models (Mixtral, DeepSeek, OLMo, Phi).
- Positive: `apply_to_qk()` default no-op means zero changes to existing GPT-2 code.
- Negative: Some code duplication between TransformerBlock and LlamaTransformerBlock (same pre-norm residual structure, different internals).
- Negative: Adding `apply_to_qk` to the ABC means every custom PE subclass must at minimum consider the interface, even if they don't need it.
