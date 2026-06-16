"""Decoupled training for the LookaheadSparseIndexer.

Implements the paper's backbone-free training strategy (Section 2.3):
1. Run frozen backbone on long documents to pre-compute hidden states + compressed KV
2. Cross-layer majority voting for golden labels (Section 2.2)
3. Train query projections with BCE → Focal Loss, negative sampling 3:1

Usage — with pre-computed data (recommended):
    uv run python docs/experiments/lookahead-sparse-attention/generate_training_data.py
    uv run python docs/experiments/lookahead-sparse-attention/train_indexer.py \\
        --data-path precomputed_data/training_data_*.pt

Usage — synthetic data (debug):
    uv run python docs/experiments/lookahead-sparse-attention/train_indexer.py

The backbone model is NEVER loaded during training — only pre-computed tensors.
"""

from __future__ import annotations

import argparse
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def focal_loss(
    preds: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
) -> torch.Tensor:
    bce = F.binary_cross_entropy(preds, targets.float(), reduction="none")
    pt = preds * targets + (1 - preds) * (1 - targets)
    focal_weight = (1 - pt) ** gamma
    return (focal_weight * bce).mean()


class PrecomputedIndexerDataset(Dataset):
    """Dataset of pre-computed hidden states and compressed KV keys.

    Each sample corresponds to one lookahead evaluation point (every τ tokens).
    """

    def __init__(
        self,
        samples: list[dict],
        frozen_keys: dict[int, torch.Tensor],
        neg_sample_ratio: float = 3.0,
    ):
        self.samples = samples
        self.frozen_keys = frozen_keys
        self.neg_sample_ratio = neg_sample_ratio

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        hidden = sample["hidden_states"]
        keys = sample["compressed_keys"]
        golden = sample["golden_labels"]
        all_chunks = sample["chunk_ids"]

        pos_chunks = [c for c in all_chunks if c in golden]
        neg_pool = [c for c in all_chunks if c not in golden]

        n_pos = len(pos_chunks)
        n_neg = min(int(n_pos * self.neg_sample_ratio), len(neg_pool)) if n_pos > 0 else min(10, len(neg_pool))
        neg_chunks = random.sample(neg_pool, n_neg) if n_neg > 0 else []

        sampled_chunks = pos_chunks + neg_chunks
        labels = [1] * len(pos_chunks) + [0] * len(neg_chunks)

        keys_list = [keys[c].clone() for c in sampled_chunks]

        return {
            "hidden_states": hidden,
            "compressed_keys": keys_list,
            "chunk_ids": sampled_chunks,
            "labels": torch.tensor(labels, dtype=torch.float32),
        }


def collate_indexer_batch(batch: list[dict]) -> dict:
    return {
        "hidden_states": [b["hidden_states"] for b in batch],
        "chunk_data": [
            {"keys": b["compressed_keys"], "chunk_ids": b["chunk_ids"], "labels": b["labels"]}
            for b in batch
        ],
    }


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    gamma: float = 2.0,
    device: str = "cpu",
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in tqdm(dataloader, desc="Training"):
        optimizer.zero_grad()

        batch_loss = 0.0
        n_total_labels = 0
        for hidden_dict, chunk_info in zip(batch["hidden_states"], batch["chunk_data"]):
            for cid, k_frozen in zip(chunk_info["chunk_ids"], chunk_info["keys"]):
                if cid not in model._frozen_keys:
                    model._frozen_keys[cid] = k_frozen.to(device)
            model._initialized = True

            pred_scores = model.score(hidden_dict, list(set(chunk_info["chunk_ids"])))

            vs = []
            for cid in chunk_info["chunk_ids"]:
                s = pred_scores.get(cid)
                if s is None:
                    s = torch.zeros(1, device=device)
                vs.append(s.view(-1))
            pred_tensor = torch.cat(vs)
            loss = focal_loss(pred_tensor, chunk_info["labels"].to(device), gamma=gamma)
            batch_loss = batch_loss + loss
            n_total_labels += len(chunk_info["chunk_ids"])

        batch_loss = batch_loss / max(n_total_labels, 1)
        batch_loss.backward()
        optimizer.step()

        total_loss += batch_loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def load_precomputed_data(data_path: str) -> tuple[list[dict], dict[int, torch.Tensor], dict]:
    data = torch.load(data_path, weights_only=True)
    return data["samples"], data["frozen_keys"], data["metadata"]


def generate_synthetic_data(
    d_model: int,
    kv_lora_rank: int,
    indexer_layers: list[int],
    n_samples: int,
    n_chunks_per_sample: int,
    device: str,
) -> tuple[list[dict], dict[int, torch.Tensor]]:
    samples: list[dict] = []
    frozen_keys: dict[int, torch.Tensor] = {}
    next_cid = 0

    for _ in range(n_samples):
        hidden_states = {
            lid: torch.randn(d_model, device=device)
            for lid in indexer_layers
        }
        chunk_ids = list(range(n_chunks_per_sample))
        compressed_keys = {}
        for c in chunk_ids:
            cid = next_cid
            compressed_keys[c] = torch.randn(1, kv_lora_rank, device=device)
            frozen_keys[cid] = compressed_keys[c]
            next_cid += 1

        golden = set(random.sample(chunk_ids, k=max(1, n_chunks_per_sample // 5)))

        samples.append({
            "hidden_states": hidden_states,
            "compressed_keys": compressed_keys,
            "golden_labels": golden,
            "chunk_ids": chunk_ids,
        })

    return samples, frozen_keys


def main():
    parser = argparse.ArgumentParser(description="Train LookaheadSparseIndexer")
    parser.add_argument("--data-path", default=None, help="Path to pre-computed .pt file")
    parser.add_argument("--model-type", default="llama")
    parser.add_argument("--max-seq-len", type=int, default=16384)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--neg-ratio", type=float, default=3.0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    print("=" * 60)
    print("LookaheadSparseIndexer — Decoupled Training")
    print("=" * 60)

    device = args.device

    if args.data_path is not None:
        # ---------- Load real pre-computed data ----------
        print(f"Loading pre-computed data from {args.data_path}")
        samples, frozen_keys, metadata = load_precomputed_data(args.data_path)
        d_model = metadata["d_model"]
        kv_lora_rank = metadata["kv_lora_rank"]
        n_layers = metadata["n_layers"]
        indexer_layers = metadata["indexer_layers"]
        n_samples = len(samples)
        print(f"  {n_samples} samples, {len(frozen_keys)} frozen keys")
        print(f"  d_model={d_model}, kv_lora_rank={kv_lora_rank}, n_layers={n_layers}")
        print(f"  Indexer layers: {indexer_layers}")
    else:
        # ---------- Synthetic fallback ----------
        print("No --data-path provided; using synthetic data.")
        d_model = 2048
        n_layers = 22
        kv_lora_rank = 512
        indexer_layers = [10, 12, 20]
        n_samples = 100
        n_chunks = 20
        samples, frozen_keys = generate_synthetic_data(
            d_model, kv_lora_rank, indexer_layers, n_samples, n_chunks, device
        )
        print(f"  {n_samples} synthetic samples, {len(frozen_keys)} frozen keys")

    print(f"  Device: {device}")
    print()

    # ---------- Build indexer ----------
    from snapmind.layers.indexer.lookahead import LookaheadSparseIndexer

    indexer = LookaheadSparseIndexer(
        d_model=d_model,
        n_layers=n_layers,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=64,
        q_lora_rank=d_model,
        indexer_layers=indexer_layers,
        n_indexer_heads=4,
    ).to(device)

    # Pre-populate frozen keys
    for cid, k in frozen_keys.items():
        indexer._frozen_keys[cid] = k.to(device)
    indexer._initialized = True

    optimizer = torch.optim.AdamW(indexer.parameters(), lr=args.lr)

    # ---------- Build dataset ----------
    dataset = PrecomputedIndexerDataset(
        samples=samples,
        frozen_keys=frozen_keys,
        neg_sample_ratio=args.neg_ratio,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_indexer_batch,
    )

    print(f"Dataset: {len(dataset)} samples")
    print(f"Indexer parameters: {sum(p.numel() for p in indexer.parameters())}")
    print()

    # ---------- Training loop ----------
    for epoch in range(args.epochs):
        loss = train_epoch(indexer, dataloader, optimizer, gamma=args.gamma, device=device)
        print(f"Epoch {epoch + 1}/{args.epochs} — loss: {loss:.6f}")

    save_path = "lookahead_indexer.pt"
    torch.save(indexer.state_dict(), save_path)
    print(f"\nIndexer weights saved to {save_path}")
    print("Done.")


if __name__ == "__main__":
    main()
