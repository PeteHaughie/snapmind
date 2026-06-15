# MLA — Notes

## Why not yet implemented
MLA's design is tightly coupled with DeepSeek-V2's MoE architecture and 200B+ scale. The memory savings (latent dimension 512 vs. ~16K full KV) matter at that scale but are less impactful on sub-3B models where the entire KV cache for 32K tokens is ~200 MB. The extra up-projection FLOPs on each decode step would increase latency for single-user interactive use — the opposite of our goals.

## What we would need
1. `MLAAttentionConfig` in `config.py`.
2. `MLAAttention` class — down-projection `W_dkv`, up-projections `W_uk` / `W_uv`.
3. Decoupled RoPE path (separate down/up for the rotary component).
4. A new `cache/mla.py` KV cache that stores `(c_kv, position)` pairs instead of `(k, v)` buffers.
5. A `MLAModel` class that uses `MLAAttention` instead of `GroupedQueryAttention`.
6. Weight loader remaps for DeepSeek-V2 style naming.
7. Perplexity validation on a held-out set — random weights won't verify correctness.

## Implementation sketch (forward)
```python
def forward(self, x, cache, position):
    # Compress: c_kv = self.W_dkv(torch.cat([x @ Wk, x @ Wv], dim=-1))
    c_kv = self.down_proj(self._concat_kv(x))
    cache.update(position, c_kv)

    # Decompress at attention time
    k = self.up_proj_k(c_kv).view(batch, n_heads, seq_len, head_dim)
    v = self.up_proj_v(c_kv).view(batch, n_kv_heads, seq_len, head_dim)
    # Apply RoPE to k, then standard GQA
```

## Future trigger
Revisit when (a) we target 7B+ models where KV cache exceeds 1 GB, (b) a pure-MLA model (non-MoE) is released with open weights, or (c) we add CUDA support and the extra projections become cheap relative to memory savings.
