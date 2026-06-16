# ─── SECTION: Lookahead Sparse Indexer ──────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F

from snapmind.core.registry import INDEXER
from snapmind.layers.indexer.base import IndexerABC


# ANCHOR: LookaheadSparseIndexer
@INDEXER.register("lookahead_sparse")
class LookaheadSparseIndexer(nn.Module, IndexerABC):
    def __init__(
        self,
        d_model: int,
        n_layers: int,
        kv_lora_rank: int = 512,
        qk_rope_head_dim: int = 64,
        q_lora_rank: int = 2048,
        indexer_layers: list[int] | None = None,
        n_indexer_heads: int = 4,
    ):
        nn.Module.__init__(self)
        self.d_model = d_model
        self.n_layers = n_layers
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.q_lora_rank = q_lora_rank
        self.indexer_layers = indexer_layers or [10, 12, 20]
        self.n_indexer_heads = n_indexer_heads

        self.q_down_proj = nn.Linear(d_model, kv_lora_rank, bias=False)
        indexer_out_dim = n_indexer_heads * kv_lora_rank
        self.q_up_proj = nn.Linear(kv_lora_rank, indexer_out_dim, bias=False)
        self.w_proj = nn.Linear(d_model, n_indexer_heads, bias=False)

        self._frozen_keys: dict[int, torch.Tensor] = {}
        self._initialized = False

    def build_frozen_keys(
        self,
        key_states: dict[int, torch.Tensor],
        chunk_ids: list[int],
    ) -> None:
        for cid in chunk_ids:
            k = key_states.get(cid)
            if k is not None:
                latent = k[..., : self.kv_lora_rank]
                self._frozen_keys[cid] = latent.mean(dim=-2, keepdim=True)
        self._initialized = True

    def score(
        self,
        hidden_states: dict[int, torch.Tensor],
        chunk_ids: list[int],
    ) -> dict[int, torch.Tensor]:
        if not self._initialized or not chunk_ids:
            return {cid: torch.tensor(0.5) for cid in chunk_ids}

        scores: dict[int, torch.Tensor] = {}

        available_layers = [lid for lid in self.indexer_layers if lid in hidden_states]
        if not available_layers:
            fallback_layer = min(hidden_states.keys())
            for cid in chunk_ids:
                scores[cid] = self._score_single(
                    hidden_states[fallback_layer].squeeze(0),
                    cid,
                )
            return scores

        agg_hidden = torch.stack(
            [hidden_states[lid].squeeze(0) for lid in available_layers],
            dim=0,
        ).mean(dim=0)

        for cid in chunk_ids:
            scores[cid] = self._score_single(agg_hidden, cid)

        return scores

    def _score_single(self, h: torch.Tensor, chunk_id: int) -> torch.Tensor:
        if chunk_id not in self._frozen_keys:
            return torch.tensor(0.0, device=h.device)
        batch_h = h.unsqueeze(0) if h.dim() == 1 else h
        q_down = self.q_down_proj(batch_h)
        q_up = self.q_up_proj(q_down)

        n_heads = self.n_indexer_heads
        q_heads = q_up.view(-1, n_heads, self.kv_lora_rank)

        w = self.w_proj(batch_h)
        w_weights = w.view(-1, n_heads)

        k_frozen = self._frozen_keys[chunk_id].to(batch_h.device)

        head_scores = []
        for h_idx in range(n_heads):
            q_h = q_heads[:, h_idx, :]
            w_h = w_weights[:, h_idx].unsqueeze(-1)
            raw = torch.mm(q_h, k_frozen.transpose(-2, -1))
            activated = F.relu(raw) * w_h
            head_scores.append(activated)

        fused = torch.stack(head_scores).sum(dim=0)
        return torch.sigmoid(fused).mean()


# ENDANCHOR: LookaheadSparseIndexer
# ─── ENDSECTION: Lookahead Sparse Indexer ──────────────
