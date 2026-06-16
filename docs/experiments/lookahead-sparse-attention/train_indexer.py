"""Decoupled training for the LookaheadSparseIndexer.

Implements the paper's backbone-free training strategy (Section 2.3):
1. Run frozen backbone on long documents to pre-compute hidden states + compressed KV
2. Cross-layer majority voting for golden labels (Section 2.2)
3. Train query projections with BCE → Focal Loss, negative sampling 3:1

Usage:
    uv run python docs/experiments/lookahead-sparse-attention/train_indexer.py \
        --model-type llama --max-seq-len 16384 --epochs 5

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
    """Focal Loss with BCE base.

    Args:
        preds: Sigmoid-activated scores in (0, 1), shape (N,).
        targets: Binary labels {0, 1}, shape (N,).
        gamma: Focusing parameter (default 2.0 per paper).

    Returns:
        Scalar loss.
    """
    bce = F.binary_cross_entropy(preds, targets.float(), reduction="none")
    pt = preds * targets + (1 - preds) * (1 - targets)
    focal_weight = (1 - pt) ** gamma
    return (focal_weight * bce).mean()


def cross_layer_majority_voting(
    logit_scores: dict[int, torch.Tensor],
    n_layers: int,
    p_threshold: float = 0.6,
    vote_threshold: int = 3,
) -> set[int]:
    """Cross-layer majority voting for golden label filtering (paper Section 2.2).

    Args:
        logit_scores: ``{layer_idx: (n_chunks,)}`` raw indexer logits.
        n_layers: Total number of CSA layers.
        p_threshold: Nucleus threshold for top-p filtering (default 0.6).
        vote_threshold: Minimum layer votes for golden label (default 3).

    Returns:
        Set of chunk IDs that are golden entries.
    """
    layer_votes: dict[int, set[int]] = {}
    for layer_idx in range(n_layers):
        scores = logit_scores.get(layer_idx)
        if scores is None:
            continue
        probs = F.softmax(scores, dim=0)
        sorted_vals, sorted_idx = probs.sort(descending=True)
        cumsum = sorted_vals.cumsum(dim=0)
        cutoff_mask = cumsum <= p_threshold
        if not cutoff_mask.any():
            cutoff_mask[0] = True
        selected = set(sorted_idx[cutoff_mask].tolist())
        layer_votes[layer_idx] = selected

    chunk_votes: dict[int, int] = {}
    for layer_selected in layer_votes.values():
        for cid in layer_selected:
            chunk_votes[cid] = chunk_votes.get(cid, 0) + 1

    golden = {cid for cid, votes in chunk_votes.items() if votes >= vote_threshold}
    return golden


class PrecomputedIndexerDataset(Dataset):
    """Dataset of pre-computed hidden states and compressed KV keys.

    Each sample corresponds to one lookahead evaluation point (every τ tokens).
    """

    def __init__(
        self,
        hidden_states: list[dict[int, torch.Tensor]],
        compressed_keys: list[dict[int, torch.Tensor]],
        golden_labels: list[set[int]],
        chunk_ids_list: list[list[int]],
        neg_sample_ratio: float = 3.0,
    ):
        self.hidden_states = hidden_states
        self.compressed_keys = compressed_keys
        self.golden_labels = golden_labels
        self.chunk_ids_list = chunk_ids_list
        self.neg_sample_ratio = neg_sample_ratio

    def __len__(self) -> int:
        return len(self.hidden_states)

    def __getitem__(self, idx: int) -> dict:
        hidden = self.hidden_states[idx]
        keys = self.compressed_keys[idx]
        golden = self.golden_labels[idx]
        all_chunks = self.chunk_ids_list[idx]

        pos_chunks = list(golden & set(all_chunks))
        neg_pool = [c for c in all_chunks if c not in golden]

        n_pos = len(pos_chunks)
        n_neg = min(int(n_pos * self.neg_sample_ratio), len(neg_pool)) if n_pos > 0 else min(10, len(neg_pool))
        neg_chunks = random.sample(neg_pool, n_neg) if n_neg > 0 else []

        sampled_chunks = pos_chunks + neg_chunks
        labels = [1] * len(pos_chunks) + [0] * len(neg_chunks)

        keys_list = []
        for c in sampled_chunks:
            if c in keys:
                keys_list.append(keys[c])
            else:
                keys_list.append(torch.zeros(1, 512))

        return {
            "hidden_states": hidden,
            "compressed_keys": keys_list,
            "chunk_ids": sampled_chunks,
            "labels": torch.tensor(labels, dtype=torch.float32),
        }


def collate_indexer_batch(batch: list[dict]) -> dict:
    hidden_list = [b["hidden_states"] for b in batch]
    all_keys = [k for b in batch for k in b["compressed_keys"]]
    all_labels = torch.cat([b["labels"] for b in batch])
    all_chunk_ids = [c for b in batch for c in b["chunk_ids"]]
    return {
        "hidden_states": hidden_list,
        "compressed_keys": all_keys,
        "chunk_ids": all_chunk_ids,
        "labels": all_labels,
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
        for hidden_dict, keys_list, chunk_ids, labels in zip(
            batch["hidden_states"],
            batch["compressed_keys"],
            batch["chunk_ids"],
            batch["labels"],
        ):
            model.zero_grad()
            for cid, k_frozen in zip(chunk_ids, keys_list):
                if cid not in model._frozen_keys:
                    model._frozen_keys[cid] = k_frozen.to(device)
            model._initialized = True

            pred_scores = model.score(hidden_dict, list(set(chunk_ids)))

            pred_tensor = torch.tensor(
                [pred_scores.get(cid, 0.0) for cid in chunk_ids],
                device=device,
            )
            loss = focal_loss(pred_tensor, labels.to(device), gamma=gamma)
            batch_loss = batch_loss + loss

        batch_loss = batch_loss / len(batch["labels"])
        batch_loss.backward()
        optimizer.step()

        total_loss += batch_loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def main():
    parser = argparse.ArgumentParser(description="Train LookaheadSparseIndexer")
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
    print(f"Model type: {args.model_type}")
    print(f"Max seq len: {args.max_seq_len}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Focal gamma: {args.gamma}")
    print(f"Negative ratio: {args.neg_ratio}")
    print(f"Device: {args.device}")
    print()

    dummy_config = {
        "d_model": 2048,
        "n_layers": 22,
        "kv_lora_rank": 512,
        "qk_rope_head_dim": 64,
        "q_lora_rank": 2048,
        "indexer_layers": [10, 12, 20],
        "n_indexer_heads": 4,
    }

    from snapmind.layers.indexer.lookahead import LookaheadSparseIndexer

    indexer = LookaheadSparseIndexer(
        d_model=dummy_config["d_model"],
        n_layers=dummy_config["n_layers"],
        kv_lora_rank=dummy_config["kv_lora_rank"],
        qk_rope_head_dim=dummy_config["qk_rope_head_dim"],
        q_lora_rank=dummy_config["q_lora_rank"],
        indexer_layers=dummy_config["indexer_layers"],
        n_indexer_heads=dummy_config["n_indexer_heads"],
    ).to(args.device)

    optimizer = torch.optim.AdamW(indexer.parameters(), lr=args.lr)

    n_hidden = 100
    n_chunks_per_sample = 20
    hidden_states_list: list[dict[int, torch.Tensor]] = []
    compressed_keys_list: list[dict[int, torch.Tensor]] = []
    golden_labels_list: list[set[int]] = []
    chunk_ids_list: list[list[int]] = []

    for _ in range(n_hidden):
        hs = {
            lid: torch.randn(dummy_config["d_model"], device=args.device)
            for lid in dummy_config["indexer_layers"]
        }
        hidden_states_list.append(hs)

        ck = {}
        chunk_ids = list(range(n_chunks_per_sample))
        for c in chunk_ids:
            ck[c] = torch.randn(1, dummy_config["kv_lora_rank"], device=args.device)
        compressed_keys_list.append(ck)
        chunk_ids_list.append(chunk_ids)

        golden = set(random.sample(chunk_ids, k=max(1, n_chunks_per_sample // 5)))
        golden_labels_list.append(golden)

    dataset = PrecomputedIndexerDataset(
        hidden_states=hidden_states_list,
        compressed_keys=compressed_keys_list,
        golden_labels=golden_labels_list,
        chunk_ids_list=chunk_ids_list,
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

    for epoch in range(args.epochs):
        loss = train_epoch(indexer, dataloader, optimizer, gamma=args.gamma, device=args.device)
        print(f"Epoch {epoch + 1}/{args.epochs} — loss: {loss:.6f}")

    save_path = "lookahead_indexer.pt"
    torch.save(indexer.state_dict(), save_path)
    print(f"\nIndexer weights saved to {save_path}")
    print("Done.")


if __name__ == "__main__":
    main()
