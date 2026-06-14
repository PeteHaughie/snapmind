import pytest
import torch


class TestPrefill:
    def test_prefill_returns_logits_and_ttft(self, tiny_gpt2, test_tokens):
        from snapmind.engine.prefill import prefill
        kv_cache = {i: {"k": None, "v": None} for i in range(tiny_gpt2.config.n_layers)}
        logits, ttft = prefill(tiny_gpt2, test_tokens, kv_cache)
        assert logits.shape == (2, 256)
        assert isinstance(ttft, float)
        assert ttft >= 0

    def test_prefill_populates_kv_cache(self, tiny_gpt2, test_tokens):
        from snapmind.engine.prefill import prefill
        n = tiny_gpt2.config.n_layers
        kv_cache = {i: {"k": None, "v": None} for i in range(n)}
        prefill(tiny_gpt2, test_tokens, kv_cache)
        for i in range(n):
            assert kv_cache[i]["k"] is not None
            assert kv_cache[i]["v"] is not None

    def test_prefill_kv_cache_shapes(self, tiny_gpt2, test_tokens):
        from snapmind.engine.prefill import prefill
        cfg = tiny_gpt2.config
        kv_cache = {i: {"k": None, "v": None} for i in range(cfg.n_layers)}
        prefill(tiny_gpt2, test_tokens, kv_cache)
        batch, seq_len = test_tokens.shape
        head_dim = cfg.d_model // cfg.n_heads
        for i in range(cfg.n_layers):
            assert kv_cache[i]["k"].shape == (batch, cfg.n_heads, seq_len, head_dim)
            assert kv_cache[i]["v"].shape == (batch, cfg.n_heads, seq_len, head_dim)

    def test_prefill_is_deterministic(self, tiny_gpt2, test_tokens):
        from snapmind.engine.prefill import prefill
        kv_cache_1 = {i: {"k": None, "v": None} for i in range(tiny_gpt2.config.n_layers)}
        kv_cache_2 = {i: {"k": None, "v": None} for i in range(tiny_gpt2.config.n_layers)}
        logits_1, _ = prefill(tiny_gpt2, test_tokens, kv_cache_1)
        logits_2, _ = prefill(tiny_gpt2, test_tokens, kv_cache_2)
        assert torch.allclose(logits_1, logits_2, atol=1e-6)

    def test_prefill_ttft_positive(self, tiny_gpt2, test_tokens):
        from snapmind.engine.prefill import prefill
        kv_cache = {i: {"k": None, "v": None} for i in range(tiny_gpt2.config.n_layers)}
        _, ttft = prefill(tiny_gpt2, test_tokens, kv_cache)
        assert ttft > 0
