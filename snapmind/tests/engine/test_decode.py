import pytest
import torch


@pytest.fixture
def setup_decode(tiny_gpt2):
    cfg = tiny_gpt2.config
    tokens = torch.randint(0, cfg.vocab_size, (1, 8))
    kv_cache = {i: {"k": None, "v": None} for i in range(cfg.n_layers)}
    tiny_gpt2(tokens, kv_cache=kv_cache)
    return tiny_gpt2, kv_cache


class TestDecodeStep:
    def test_decode_returns_int_token(self, setup_decode):
        from snapmind.engine.decode import decode_step
        from snapmind.sampling.greedy import GreedySampler

        model, kv_cache = setup_decode
        token_id = decode_step(model, 42, kv_cache, GreedySampler())
        assert isinstance(token_id, int)
        assert 0 <= token_id < model.config.vocab_size

    def test_decode_extends_kv_cache(self, setup_decode):
        from snapmind.engine.decode import decode_step
        from snapmind.sampling.greedy import GreedySampler

        model, kv_cache = setup_decode
        orig_len = kv_cache[0]["k"].shape[-2]
        decode_step(model, 42, kv_cache, GreedySampler())
        assert kv_cache[0]["k"].shape[-2] == orig_len + 1

    def test_decode_deterministic_greedy(self, setup_decode):
        from snapmind.engine.decode import decode_step
        from snapmind.sampling.greedy import GreedySampler

        model1, kv1 = setup_decode
        model2, kv2 = setup_decode
        # rebuild separate kv caches
        cfg = model1.config
        kv1 = {i: {"k": kv1[i]["k"].clone(), "v": kv1[i]["v"].clone()} for i in range(cfg.n_layers)}
        kv2 = {i: {"k": kv2[i]["k"].clone(), "v": kv2[i]["v"].clone()} for i in range(cfg.n_layers)}
        t1 = decode_step(model1, 42, kv1, GreedySampler())
        t2 = decode_step(model2, 42, kv2, GreedySampler())
        assert t1 == t2

    def test_decode_all_tokens_finite(self, setup_decode):
        from snapmind.engine.decode import decode_step
        from snapmind.sampling.greedy import GreedySampler

        model, kv_cache = setup_decode
        tokens = []
        for _ in range(5):
            t = decode_step(model, 42, kv_cache, GreedySampler())
            tokens.append(t)
        assert all(isinstance(t, int) for t in tokens)
