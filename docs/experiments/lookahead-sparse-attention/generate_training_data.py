"""Generate pre-computed training data for LookaheadSparseIndexer.

Pipeline:
1. Load backbone (TinyLlama 1.1B with real HF weights)
2. Load PG-19 dataset
3. For each document → tokenize → forward pass → extract per-chunk
   attention scores + hidden states + compressed K keys
4. Cross-layer majority voting → golden labels
5. Save to disk for decoupled training

Usage:
    uv run python docs/experiments/lookahead-sparse-attention/generate_training_data.py
"""

from __future__ import annotations

import argparse
import os
import time
from collections.abc import Callable

import torch
import torch.nn.functional as F
from tqdm import tqdm

from snapmind.core.architecture import ARCHITECTURE
from snapmind.core.config import ModelConfig
from snapmind.loaders.safetensors import SafetensorsLoader
from snapmind.tokenizer.hf import HFTokenizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CHUNK_SIZE = 64  # paper's τ interval (tokens per chunk)
INDEXER_LAYERS = [10, 12, 20]  # paper's Pareto optimum
P_THRESHOLD = 0.6  # top-p nucleus threshold
VOTE_THRESHOLD = 3  # cross-layer majority threshold
MAX_SEQ_LEN = 2048  # TinyLlama context window
STRIDE = 1024  # overlap between consecutive windows
REF_LAYER_FOR_KEYS = 10  # which layer's K to use for compressed keys


def compute_golden_labels(
    per_layer_scores: dict[int, torch.Tensor],
    n_layers: int,
    t: int,
    seq_len: int,
    p_threshold: float = P_THRESHOLD,
    vote_threshold: int = VOTE_THRESHOLD,
) -> set[int]:
    """Cross-layer majority voting for the lookahead window starting at *t*.

    Args:
        per_layer_scores: ``{layer_idx: (seq_len, n_chunks)}`` — per-chunk
            attention scores (our proxy for the paper's raw indexer logits).
        n_layers: Total number of layers in the backbone.
        t: Start position of the lookahead window (must be a τ boundary).
        seq_len: Sequence length for this window.
        p_threshold: Top-p threshold for layer-level selection.
        vote_threshold: Minimum layer votes for a golden entry.

    Returns:
        Set of chunk IDs that are golden for this evaluation point.
    """
    n_chunks_before = t // CHUNK_SIZE
    window_end = min(t + CHUNK_SIZE, seq_len)

    votes: dict[int, int] = {}

    for layer_idx in range(n_layers):
        scores = per_layer_scores.get(layer_idx)
        if scores is None:
            continue
        # scores shape: (seq_len, n_chunks); extract lookahead window rows
        window_scores = scores[t:window_end, :n_chunks_before]  # (τ, n_chunks_before)
        if window_scores.numel() == 0:
            continue

        window_probs = F.softmax(window_scores, dim=-1)
        sorted_vals, sorted_idx = window_probs.sort(descending=True, dim=-1)
        cumsum = sorted_vals.cumsum(dim=-1)

        for qi in range(window_end - t):
            mask = cumsum[qi] <= p_threshold
            if not mask.any():
                mask[0] = True
            for cid in sorted_idx[qi, mask].tolist():
                if cid < n_chunks_before:
                    votes[cid] = votes.get(cid, 0) + 1

    return {cid for cid, v in votes.items() if v >= vote_threshold}


def build_frozen_keys(
    kv_cache: list[dict],
    n_chunks: int,
    n_kv_heads: int,
    head_dim: int,
) -> dict[int, torch.Tensor]:
    """Compute compressed KV keys from the reference layer's K cache.

    Each compressed key is the mean-pooled K over positions in a chunk,
    flattened to ``(n_kv_heads * head_dim,)``.

    Returns:
        ``{chunk_id: tensor (1, kv_lora_rank)}``
    """
    k = kv_cache[REF_LAYER_FOR_KEYS]["k"]  # (1, n_kv_heads, seq_len, head_dim)
    keys: dict[int, torch.Tensor] = {}
    for c in range(n_chunks):
        start = c * CHUNK_SIZE
        end = min(start + CHUNK_SIZE, k.shape[-2])
        chunk_k = k[0, :, start:end, :].mean(dim=1).reshape(-1)  # (n_kv_heads * head_dim,)
        keys[c] = chunk_k.unsqueeze(0)  # (1, kv_lora_rank)
    return keys


def main():
    parser = argparse.ArgumentParser(description="Generate PG-19 training data for LookaheadSparseIndexer")
    parser.add_argument("--max-books", type=int, default=5, help="Number of PG-19 books to process")
    parser.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LEN)
    parser.add_argument("--stride", type=int, default=STRIDE)
    parser.add_argument("--output-dir", default="precomputed_data")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype) if isinstance(args.dtype, str) else args.dtype

    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load backbone model + tokenizer
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Step 1: Loading TinyLlama backbone + tokenizer")
    print("=" * 60)

    arch = ARCHITECTURE.get("tinyllama")
    config = ModelConfig(**arch.default_config)
    config.model_type = "tinyllama"

    model = arch.model_cls(config)
    model.eval()

    tokenizer = HFTokenizer("tinyllama")

    print(f"  Loading weights from {arch.hf_repo} ...")
    loader = SafetensorsLoader()
    result = loader.load(None, model, config)
    if result["missing"]:
        print(f"  Missing keys: {result['missing']}")
    if result["unexpected"]:
        print(f"  Unexpected keys: {result['unexpected']}")
    print(f"  Model parameter count: {sum(p.numel() for p in model.parameters()):,}")

    model = model.to(device=device, dtype=dtype)
    n_layers = config.n_layers
    n_kv_heads = config.n_kv_heads
    head_dim = config.d_model // config.n_heads
    kv_lora_rank = n_kv_heads * head_dim
    print(f"  n_layers={n_layers}, n_kv_heads={n_kv_heads}, head_dim={head_dim}, kv_lora_rank={kv_lora_rank}")

    # ------------------------------------------------------------------
    # 2. Load PG-19 dataset
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Step 2: Loading PG-19 dataset")
    print("=" * 60)

    from datasets import load_dataset

    ds = load_dataset("emozilla/pg19", split="train", streaming=True)
    print("  PG-19 train set (streaming)")

    long_enough: list = []
    i = -1
    for i, example in enumerate(ds):
        if len(long_enough) >= args.max_books:
            break
        if example["text"] and len(example["text"]) > args.max_seq_len:
            long_enough.append(example)

    books = long_enough
    print(f"  Processing {len(books)} books (scanned {i + 1} total)")

    # Truncate each book to avoid excessive processing
    # A single book can be 500K+ chars; limit to 5 windows per book
    max_tokens_per_book = args.max_seq_len * 5
    for bk in books:
        tokens = tokenizer.encode(bk["text"])
        if len(tokens) > max_tokens_per_book:
            bk["text_trunc"] = tokenizer.decode(tokens[:max_tokens_per_book])
        else:
            bk["text_trunc"] = bk["text"]

    # ------------------------------------------------------------------
    # 3. Process documents → extract training samples
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Step 3: Forward passes + data extraction")
    print("=" * 60)

    all_samples: list[dict] = []
    all_frozen_keys: dict[int, torch.Tensor] = {}
    next_global_chunk_id = 0

    for book_idx, book in enumerate(tqdm(books, desc="Books")):
        text = book["text_trunc"]

        # Tokenize
        input_ids = tokenizer.encode(text)

        if len(input_ids) < args.max_seq_len:
            continue

        # Slide windows over the document
        for window_start in range(0, len(input_ids) - args.max_seq_len + 1, args.stride):
            window_ids = input_ids[window_start : window_start + args.max_seq_len]
            tokens = torch.tensor([window_ids], device=device)  # (1, seq_len)

            seq_len = tokens.shape[1]
            n_chunks = (seq_len + CHUNK_SIZE - 1) // CHUNK_SIZE

            # --- Set up hooks ---
            layer_outputs: dict[int, torch.Tensor] = {}
            per_layer_scores: dict[int, torch.Tensor] = {}

            def make_block_hook(lid: int) -> Callable:
                def hook(_module, _input, output):
                    layer_outputs[lid] = output.detach().clone().cpu()
                return hook

            def make_attn_hook(lid: int, cs: int) -> Callable:
                def hook(_module, _input, output):
                    _, attn_weights = output  # (1, n_heads, seq_len, seq_len)
                    attn_weights = attn_weights.cpu()  # move to CPU early
                    sl = attn_weights.shape[-1]
                    nc = (sl + cs - 1) // cs
                    scores = torch.zeros(sl, nc)
                    for s in range(nc):
                        s_start = s * cs
                        s_end = min(s_start + cs, sl)
                        scores[:, s] = attn_weights[0, :, :, s_start:s_end].sum(dim=(0, 2))
                    per_layer_scores[lid] = scores
                return hook

            handles: list = []
            for lid in INDEXER_LAYERS:
                if lid < len(model.layers):
                    handles.append(model.layers[lid].register_forward_hook(make_block_hook(lid)))
            for lid in range(n_layers):
                handles.append(
                    model.layers[lid].self_attn.register_forward_hook(make_attn_hook(lid, CHUNK_SIZE))
                )

            # --- Forward pass ---
            kv_cache = [{"k": None, "v": None} for _ in range(n_layers)]

            with torch.no_grad():
                model(tokens, kv_cache=kv_cache)

            # --- Remove hooks ---
            for h in handles:
                h.remove()

            # --- Compute golden labels per evaluation point ---
            for t in range(CHUNK_SIZE, seq_len - CHUNK_SIZE + 1, CHUNK_SIZE):
                golden = compute_golden_labels(
                    per_layer_scores,
                    n_layers,
                    t,
                    seq_len,
                )

                # Build hidden states dict
                hidden_states: dict[int, torch.Tensor] = {}
                for lid in INDEXER_LAYERS:
                    hs = layer_outputs.get(lid)
                    if hs is not None:
                        hidden_states[lid] = hs[0, t, :].clone()

                # Build compressed keys for chunks before t
                n_chunks_before = t // CHUNK_SIZE
                compressed_keys: dict[int, torch.Tensor] = {}
                for c in range(n_chunks_before):
                    global_id = next_global_chunk_id + c
                    if global_id not in all_frozen_keys:
                        start = c * CHUNK_SIZE
                        end = min(start + CHUNK_SIZE, seq_len)
                        k = kv_cache[REF_LAYER_FOR_KEYS]["k"]  # (1, n_kv_heads, seq_len, head_dim)
                        chunk_k = k[0, :, start:end, :].mean(dim=1).reshape(-1)
                        all_frozen_keys[global_id] = chunk_k.unsqueeze(0).cpu()
                    compressed_keys[c] = all_frozen_keys[global_id]

                sample = {
                    "hidden_states": hidden_states,
                    "compressed_keys": compressed_keys,
                    "golden_labels": golden,
                    "chunk_ids": list(range(n_chunks_before)),
                }
                all_samples.append(sample)

            next_global_chunk_id += n_chunks

        print(f"  Book {book_idx + 1}: {len(all_samples)} total samples so far")

    # ------------------------------------------------------------------
    # 4. Save pre-computed data
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Step 4: Saving pre-computed data")
    print("=" * 60)

    timestamp = int(time.time())
    save_path = os.path.join(args.output_dir, f"training_data_{timestamp}.pt")
    data = {
        "samples": all_samples,
        "frozen_keys": all_frozen_keys,
        "metadata": {
            "n_samples": len(all_samples),
            "n_frozen_keys": len(all_frozen_keys),
            "kv_lora_rank": kv_lora_rank,
            "d_model": config.d_model,
            "n_layers": n_layers,
            "indexer_layers": INDEXER_LAYERS,
            "chunk_size": CHUNK_SIZE,
            "max_seq_len": args.max_seq_len,
            "stride": args.stride,
            "source": "pg19",
        },
    }
    torch.save(data, save_path)
    print(f"  Saved {len(all_samples)} samples, {len(all_frozen_keys)} frozen keys")
    print(f"  Path: {save_path}")
    print("Done.")


if __name__ == "__main__":
    main()
