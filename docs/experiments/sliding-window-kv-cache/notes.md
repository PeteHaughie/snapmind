# Sliding Window KV Cache — Notes

## Why not yet implemented
The Mistral models in our registry (tinyllama, mistral, ministral-3-3b) were trained without sliding window. Using a truncated cache on them would cause perplexity degradation beyond the window size. The benefit for our use case (single-user interactive generation on a laptop) is marginal — memory savings from a 4K window vs. 32K contiguous cache is noticeable, but we don't yet hit OOM at 32K on a 16 GB Mac.

## What we would need
1. `SlidingWindowKVCacheConfig(window_size=4096)` in `config.py`.
2. `SlidingWindowKVCache` with circular buffer and wrap-around indexing.
3. Sliding window attention mask builder — `torch.tril` with distance cutoff.
4. Evaluation script: perplexity on PG-19 vs. window size for each registered model.
5. Documentation of perplexity cliff position per model.

## Implementation sketch (attention mask)
```python
def sliding_window_mask(seq_len, window_size, device):
    mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
    for i in range(seq_len):
        lo = max(0, i - window_size + 1)
        mask[i, lo:i+1] = 0.0
    return mask
```

## Future trigger
Add when (a) we expand model support to a sliding-window-native model (Mistral v0.2+, Phi-3, Gemma 2), or (b) users report OOMs at their desired context length.
