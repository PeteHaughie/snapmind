# Paged KV Cache — Notes

## Why not yet implemented
PagedAttention was designed for CUDA GPUs where custom Triton/CUDA kernels can gather/scatter pages in a single fused op. On MPS (Apple Metal), `torch.index_add` and indexing gather ops have limited support — the fallback `for` loop over pages would dominate inference time at small batch sizes where our framework operates.

## What we would need
1. `PagedKVCacheConfig(max_blocks=1024, block_size=32)` dataclass.
2. `BlockManager` — free list of physical blocks, page table per sequence.
3. `PagedKVCache` subclass with `block_size`-aligned allocation and page-table-indexed `update()` / `get()`.
4. GQA-friendly indexing: multi-head KV needs aligned page strides per head group.
5. For MPS: benchmark whether `torch.index_select` on pages is faster than contiguous preallocation. If yes, implement; if not, reject.

## Future trigger
If Apple adds `MPSGraph` scatter/gather ops or we adopt MLX, revisit. On CUDA machines, use vLLM directly.
