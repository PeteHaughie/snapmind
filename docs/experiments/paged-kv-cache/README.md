# Paged KV Cache

## Paper Reference
Kwon, W., Li, Z., Zhuang, S., Sheng, Y., Zheng, L., Yu, C., Gonzalez, J., Zhang, H., Stoica, I. (2023). "Efficient Memory Management for Large Language Model Serving with PagedAttention." *SOSP '23*.

https://arxiv.org/abs/2309.06180

## Goal
Replace the contiguous KV cache buffer with a paged block manager that allocates fixed-size pages on demand. Eliminate internal/external fragmentation in the KV cache tensor and enable prompt prefix sharing across requests.

## Implementation Plan
1. Add `PagedKVCacheConfig(max_blocks, block_size)` to `config.py`.
2. Implement `PagedKVCache(ABC)` in a new `cache/paged.py` module.
3. Implement a `BlockManager` that allocates, frees, and shares page tables.
4. Add a `PagedKVCacheAdapter` that wraps the block manager behind the existing `KVCacheInterface`.
5. Wire it into `LlamaModel` / `MistralModel` via config option.
6. Benchmark prefill+decode throughput vs. contiguous cache.

## Design
- Each logical KV cache sequence gets a page table (list of physical block IDs).
- Attention kernel looks up `key[b]` and `value[b]` via the page table rather than contiguous slicing.
- Pages are allocated lazily on first write to a new token position.
- Block size tunable (16, 32, 64 tokens per block).
- Copy-on-write for beam search / parallel sampling sharing prefixes.

## Risks
- For small batch sizes (1–4), the page-table indirection overhead may outweigh fragmentation savings on MPS.
- Metal Performance Shaders (MPS) backend lacks scatter/gather indexed copy ops used in CUDA implementations — fallback loops will be slow.
- Without a custom MPS kernel, each attention step may need a `torch.cat` or loop to gather pages, negating memory benefits.

## Status
pending
