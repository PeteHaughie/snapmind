from collections.abc import AsyncIterator, Sequence
from typing import Any

import torch

from snapmind.core.config import EngineConfig, IndexerConfig
from snapmind.core.registry import INDEXER
from snapmind.engine.decode import decode_step
from snapmind.engine.prefill import prefill
from snapmind.kv_cache.base import KVCacheABC
from snapmind.models.base import BaseModelABC
from snapmind.sampling.base import SamplerABC
from snapmind.tokenizer.base import TokenizerABC


class GenerateEngine:
    def __init__(
        self,
        model: BaseModelABC,
        tokenizer: TokenizerABC,
        sampler: SamplerABC,
        eos_token_id: int | None = None,
        engine_config: EngineConfig | None = None,
        kvcache: KVCacheABC | None = None,
        indexer_config: IndexerConfig | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.sampler = sampler
        self._eos_token_id = eos_token_id if eos_token_id is not None else getattr(tokenizer, "eos_token_id", 50256)
        self._engine_config = engine_config or EngineConfig()
        if kvcache is not None:
            self._kvcache = kvcache
        else:
            from snapmind.kv_cache.naive import NaiveKVCache

            cfg: Any = self.model.config
            self._kvcache = NaiveKVCache(
                max_seq_len=cfg.max_seq_len,
                n_layers=cfg.n_layers,
                n_heads=cfg.n_heads,
                head_dim=cfg.d_model // cfg.n_heads,
            )

        self._indexer_config = indexer_config
        self._indexer = None
        self._hooks: list[torch.utils.hooks.RemovableHandle] = []
        self._hook_states: dict[int, torch.Tensor] = {}

        if indexer_config is not None and indexer_config.indexer_type:
            self._indexer = INDEXER.create(
                indexer_config.indexer_type,
                d_model=model.config.d_model,
                n_layers=model.config.n_layers,
                kv_lora_rank=indexer_config.kv_lora_rank,
            )
            self._register_hooks(indexer_config.indexer_layers)

    def _register_hooks(self, layer_indices: tuple[int, ...]) -> None:
        self._hook_states.clear()
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

        def make_hook(layer_idx: int):
            def hook(_module, _input, output):
                self._hook_states[layer_idx] = output[0, -1, :].detach().clone()
            return hook

        layers: Sequence[Any] = self.model.layers  # type: ignore[assignment]
        for idx in layer_indices:
            if idx < len(layers):
                handle = layers[idx].register_forward_hook(make_hook(idx))
                self._hooks.append(handle)

    def _run_indexer(self, step: int, chunk_size: int = 64) -> None:
        if self._indexer is None:
            return
        if not self._hook_states:
            return
        n_chunks = max(1, step // chunk_size)
        chunk_ids = list(range(n_chunks))
        scores = self._indexer.score(self._hook_states, chunk_ids)
        if hasattr(self._kvcache, "score_and_repage"):
            threshold = getattr(self._indexer_config, "score_threshold", 0.5)
            self._kvcache.score_and_repage(scores, threshold=threshold)

    def _make_kv_cache(self) -> dict:
        self._kvcache.reset()
        return self._kvcache.layer_dicts()

    async def generate(
        self,
        prompt: str,
        max_tokens: int | None = None,
        temperature: float = 1.0,
        **sampler_kwargs,
    ) -> AsyncIterator[str]:
        if max_tokens is None:
            max_tokens = self._engine_config.max_tokens
        kv_cache = self._make_kv_cache()
        input_ids = self.tokenizer.encode(prompt)
        device = next(self.model.parameters()).device
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

        last_logits, _ = prefill(self.model, input_tensor, kv_cache)
        sampled = self.sampler.sample(last_logits[0, :], temperature=temperature, **sampler_kwargs)
        next_token_id = int(sampled.item())

        step = len(input_ids)

        interval = getattr(self._indexer_config, "indexer_interval", 0) if self._indexer_config else 0
        if interval > 0:
            self._run_indexer(step)

        for _ in range(max_tokens):
            if next_token_id == self._eos_token_id:
                break
            token_str = self.tokenizer.decode([next_token_id])
            yield token_str
            next_token_id = decode_step(
                self.model,
                next_token_id,
                kv_cache,
                self.sampler,
                temperature=temperature,
                position_ids=torch.tensor([step], device=device),
                **sampler_kwargs,
            )
            step += 1

            if interval > 0 and step > 0 and step % interval == 0:
                self._run_indexer(step)
