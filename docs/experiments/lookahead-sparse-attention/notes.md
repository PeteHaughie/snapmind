# Implementation Diary

## 2026-06-16 — Phase 1-3 initial implementation

- Created `TieredPagedKVCache` at `snapmind/kv_cache/tiered.py`. Registered as `"tiered"`.
  - Fixed-size pool (configurable `pool_tokens`, default 4096 = 64 chunks × 64 tokens).
  - 64-token chunk granularity. Sink + trailing window always hot.
  - `score_and_repage(scores, threshold)` for pluggable indexer integration.
  - `layer_dicts()` returns per-layer dicts with only hot chunks assembled.

- Created `IndexerABC` at `snapmind/layers/indexer/base.py`. Added `INDEXER` registry to `core/registry.py`.
- Created `LookaheadSparseIndexer` at `snapmind/layers/indexer/lookahead.py`. Registered as `"lookahead_sparse"`.
  - Dual-encoder query network: q_down_proj, q_up_proj, w_proj.
  - Sigmoid-gated matching score (paper Eq. 4).
  - `build_frozen_keys()` to pre-compute compressed KV chunk representations.

- Updated `GenerateEngine` at `snapmind/engine/generate.py`:
  - Accepts optional `IndexerConfig` with `indexer_type`, `indexer_layers`, `indexer_interval`, `score_threshold`.
  - Registers forward hooks on specified layers to capture hidden states.
  - Runs indexer every τ decode steps → calls `score_and_repage()` on the KV cache.

## Open Questions

1. The per-layer dict interface in `layer_dicts()` is the bottleneck — attention layers mutate the dict directly (concatenating K/V), and the tiered cache needs to intercept this. Current approach: rebuild the dict from hot chunks on every `score_and_repage()` call. After attention appends new tokens, the next `layer_dicts()` or `fetch()` call returns the assembled hot + active view. Need to verify this round-trips correctly.

2. MLAs `kv_proj` output (k_latent + k_rope_raw) is what should be stored in the tiered cache for DeepSeek-V4-style models. For GQA/MHA, the projected K is what goes in. The tiered cache is agnostic to this — it stores whatever tensors the attention layer gives it via `store()`.

3. Training pipeline (Phase 4) needs the `pg19` dataset and a forward pass to extract hidden states + compressed keys. This is computationally heavy but only needs to run once.
