# Experiments

One folder per paper, idea, or component implementation tested.

## Directory Structure

```
docs/experiments/<short-descriptive-name>/
├── README.md          # Paper reference, status, results, rejection reason
├── notes.md           # Freeform implementation diary
└── results.tsv        # experiment / metric_before / metric_after / status
```

## Status Values

| Status | Meaning |
|---|---|
| `accepted` | Merged into the framework. Component is available by default. |
| `experimental` | Implemented but not merged. Needs more testing or review. |
| `rejected` | Tested and discarded. README says why. |
| `pending` | Planned or in progress. |

## Index

| Experiment | Status | Description |
|---|---|---|
| [Paged KV Cache](paged-kv-cache/README.md) | pending | vLLM-style page-based key/value cache to eliminate fragmentation |
| [Sliding Window KV Cache](sliding-window-kv-cache/README.md) | pending | Mistral-style fixed-size circular buffer for bounded memory |
| [MLA](mla-attention/README.md) | pending | DeepSeek-V2 Multi-head Latent Attention for compressed KV cache |
| [Lookahead Sparse Attention](lookahead-sparse-attention/README.md) | experimental | FlashMemory-style tiered KV cache + neural memory indexer for ultra-long context |
