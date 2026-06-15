# Multi-head Latent Attention (MLA)

## Paper Reference
DeepSeek-AI. (2024). "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model." *arXiv preprint arXiv:2405.04434*.

https://arxiv.org/abs/2405.04434

## Goal
Implement MLA to compress the KV cache from `2 * n_kv_heads * head_dim` per token to `2 * d_latent` per token (where `d_latent << n_kv_heads * head_dim`). In DeepSeek-V2, `d_latent` is 512 vs. ~16384 for full KV — a 32× reduction.

## Implementation Plan
1. Add `MLAAttentionConfig(latent_dim, n_heads, n_kv_heads)` to `config.py`.
2. Implement `MLAAttention(ABC)` in a new `attention/mla.py` module.
3. Down-projection: `W_dkv: (n_kv_heads * head_dim, d_latent)` compresses K and V jointly.
4. Up-projection: `W_uk: (d_latent, n_heads * head_dim)` reconstructs K for each Q head; `W_uv: (d_latent, n_kv_heads * head_dim)` reconstructs V.
5. KV cache stores only the latent vector `c_kv` (size `d_latent`) per token.
6. At attention time, up-project `c_kv` into full K and V, then compute standard multi-head attention.

## Design
- Latent concatenation: K and V share the same compressed latent `c_kv` = `concat(W_dkv * k, W_dkv * v)` or jointly as in the paper.
- Decoupled RoPE: the paper applies RoPE through an additional down/up projection path to keep the latent free of rotary interference.
- For simplicity in v1: apply RoPE to up-projected K only (not to the latent itself).
- Cached state: `(c_kv, position)` per layer — a fraction of the normal KV cache size.

## Risks
- MLA is designed for 200B+ MoE models where KV cache dominates memory. For sub-3B models in this framework, the overhead of the extra projections may exceed the memory savings.
- The up-projection at each attention step adds FLOPs proportional to `d_latent * n_heads * head_dim` — on MPS this may be significantly slower than standard attention with a small cache.
- Weight interop: DeepSeek-V2's weight naming differs substantially from the HuggingFace convention used by our loaders.
- No pre-trained publicly available weights for pure MLA (DeepSeek-V2 is MoE + MLA combined).
- Testing with random weights avoids the perplexity benchmark needed to validate correctness.

## Status
pending
