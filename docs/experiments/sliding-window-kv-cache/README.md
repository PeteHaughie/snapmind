# Sliding Window KV Cache

## Paper Reference
Jiang, A. Q., Sablayrolles, A., Mensch, A., Bamford, C., Chaplot, D. S., Casas, D. d. l., Bressand, F., Lengyel, G., Lample, G., Saulnier, L., Lavaud, L. R., Lachaux, M.-A., Stock, P., Scao, T. L., Lavril, T., Wang, T., Lacroix, T., Sayed, W. E. (2023). "Mistral 7B." *arXiv preprint arXiv:2310.06825*.

https://arxiv.org/abs/2310.06825

## Goal
Replace the unbounded contiguous KV cache with a fixed-size circular buffer that stores only the last `W` tokens (window size). Each token attends only to tokens within its window, bounding memory usage linearly with `W` regardless of sequence length.

## Implementation Plan
1. Add `SlidingWindowKVCacheConfig(window_size)` to `config.py`.
2. Implement `SlidingWindowKVCache(ABC)` in a new `cache/sliding_window.py` module.
3. Preallocate a fixed `(2, n_layers, n_kv_heads, window_size, head_dim)` buffer and wrap writes around via modulo indexing.
4. Mask attention logits so each position only sees the last `W` tokens (or fewer at the start of a sequence).
5. Wire into `LlamaModel` / `MistralModel` via config option.

## Design
- `update(position, key, value)`: store at `position % window_size` in the circular buffer.
- `get(position, num_tokens)`: return the last `min(num_tokens, window_size)` entries from the buffer, handling wrap-around.
- Attention masking: build a causal mask where position `i` can only attend to positions `[max(0, i-W+1), i]`.
- Compatible with GQA — window is per KV head, not per query head.

## Risks
- If the model was not trained with sliding window (most aren't), perplexity degrades beyond window length — the cache is a hard truncation.
- Mistral-trained models use a special `--sliding-window` attention mask at training time; other models need evaluation to find the perplexity cliff.
- Short windows (e.g., 4096) may not hold enough context for coherent generation.

## Status
pending
