# Lookahead Sparse Attention (LSA)

## Paper Reference

Wang, Y. et al. (2026). "FlashMemory-DeepSeek-V4: Lightning Index Ultra-Long Context via Lookahead Sparse Attention." *arXiv preprint arXiv:2606.09079*.

https://arxiv.org/abs/2606.09079

## Production Reference

Luce KVFlash: https://www.lucebox.com/blog/kvflash

## Goal

Implement lookahead sparse attention for snapmind's modular framework. Two cache-side and indexer-side components:

1. **TieredPagedKVCache** (`kv_cache/tiered.py`) — fixed-size GPU pool + CPU backing store. 64-token chunks are paged between GPU (hot) and CPU (cold) based on indexer scores. Attention sink (first chunk) + trailing window always pinned.

2. **LookaheadSparseIndexer** (`layers/indexer/lookahead.py`) — lightweight dual-encoder that projects hidden states from layers 10/12/20 through a low-rank query encoder and scores historical chunks via sigmoid-gated matching against frozen compressed KV keys.

## Implementation Plan

### Phase 1 — TieredPagedKVCache
- Fixed-size GPU pool (configurable via `pool_tokens`, default 4096 = 64 chunks)
- CPU backing store for cold chunks
- 64-token chunk granularity
- Sink (chunk 0) + trailing window (last 2 chunks) pinned
- `score_and_repage(scores, threshold)` method updates hot/cold split
- Registered as `"tiered"` in KV_CACHE registry

### Phase 2 — Indexer Components
- `IndexerABC` — abstract base with `score(hidden_states, chunk_ids) -> dict[int, float]`
- `LookaheadSparseIndexer` — dual-encoder: `q_down_proj` (d_model → kv_lora_rank), `q_up_proj` (kv_lora_rank → n_heads × kv_lora_rank), `w_proj` (d_model → n_heads). Sigmoid-activated gated matching score.
- Registered as `"lookahead_sparse"` in INDEXER registry

### Phase 3 — Engine Integration
- `GenerateEngine` accepts optional `IndexerConfig`
- Registers forward hooks on specified indexer layers
- Every `τ=64` decode steps: collect hidden states → `indexer.score()` → `kvcache.score_and_repage()`

### Phase 4 — Indexer Training
- Standalone decoupled training (paper section 2.3)
- Dataset: `emozilla/pg19` filtered to documents ≥16K tokens
- Data generation: forward pass on frozen backbone, extract hidden states + compressed KV keys
- Cross-layer majority voting → golden labels (softmax → top-p=0.6 → ≥3 layer consensus)
- BCE → Focal Loss (γ=2), negative sampling 3:1
- Training on CPU: only query projections trained (< 0.1% of backbone)

## Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Pool sizing | Fixed slot count (KVFlash style) | Hard memory cap vs. threshold creep in paper |
| Chunk size | 64 tokens | Matches paper's τ interval |
| Indexer layers | 10, 12, 20 | Paper's Pareto optimum across 500 runs |
| Score function | Sigmoid + ReLU gated matching | Paper's Eq. 4 |
| No GPU pool pre-allocation | Lazy device placement | PyTorch tensors can live on any device; pool limit enforced by count |
| Training device | CPU | Decoupled: backbone never loaded, only small query projections |
| Dataset | `emozilla/pg19` | 28K books, good long-document distribution |

## Risks

- **Length generalization**: Indexer trained on ≤512K context may fail at 1M+. Paper reports hard ceiling at 2× training length. Mitigation: train at max available context length.
- **No real DeepSeek-V4 weights**: The compressed key indexer (KIComp) requires MLA's kv_lora_rank latent. snapmind's MLA matches this dimensionally; testing uses random weights.
- **Dense retrieval tasks (MRCR)**: Paper reports LSA breaks on multi-range co-reference (48% vs 76%). Pool size can be increased as a fallback.
- **CPU offload overhead**: Pure PyTorch tensor `to(device)` on every repage. Acceptable at τ=64 intervals (~1% of decode time per paper).

## Status

experimental

## Results

See [results.tsv](results.tsv).
