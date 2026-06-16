import abc

import torch


class IndexerABC(abc.ABC):
    @abc.abstractmethod
    def score(self, hidden_states: dict[int, torch.Tensor], chunk_ids: list[int]) -> dict[int, float]:
        """Score each historical chunk for relevance to the current decoding window.

        Args:
            hidden_states: ``{layer_idx: (batch, d_model)}`` — hidden states from
                the query token at specific indexer layers.
            chunk_ids: Logical chunk IDs to score (one per 64-token block).

        Returns:
            ``{chunk_id: score}`` where score in (0, 1) — higher means more relevant.
        """
