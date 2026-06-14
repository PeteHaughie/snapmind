# ─── SECTION: Mistral Integration Tests ───────────────────
import pytest
import torch


@pytest.fixture
def tiny_mistral_config():
    from snapmind.core.config import ModelConfig

    return ModelConfig(
        model_type="mistral",
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        n_layers=2,
        d_ff=256,
        vocab_size=32000,
        max_seq_len=128,
        norm_eps=1e-5,
        attention_type="gqa",
        pe_type="rope",
        norm_type="rmsnorm",
        activation_type="silu",
        kv_cache_type="sliding_window",
        window_size=8,
    )


@pytest.fixture
def tiny_mistral(tiny_mistral_config):
    from snapmind.models.mistral import MistralModel

    return MistralModel(tiny_mistral_config)


@pytest.fixture
def tiny_mistral_tokenizer():
    from snapmind.tokenizer.hf import HFTokenizer

    return HFTokenizer(model_name="tinyllama")


class TestMistralIntegration:
    def test_full_pipeline_generates_tokens(self, tiny_mistral, tiny_mistral_tokenizer):
        from snapmind.engine.generate import GenerateEngine
        from snapmind.sampling.greedy import GreedySampler

        sampler = GreedySampler()
        engine = GenerateEngine(tiny_mistral, tiny_mistral_tokenizer, sampler, eos_token_id=2)

        import asyncio

        async def run():
            tokens = []
            async for token in engine.generate("Hello", max_tokens=5):
                tokens.append(token)
            return "".join(tokens)

        output = asyncio.run(run())
        assert isinstance(output, str)
        assert len(output) > 0

    def test_sliding_window_during_generation(self, tiny_mistral, tiny_mistral_config):
        kv_cache = {i: {"k": None, "v": None} for i in range(tiny_mistral_config.n_layers)}
        for seq_len in [5, 10, 15]:
            tokens = torch.randint(0, 32000, (1, seq_len))
            tiny_mistral(tokens, kv_cache=kv_cache)
            for i in range(tiny_mistral_config.n_layers):
                k = kv_cache[i]["k"]
                assert k is not None
                assert k.shape[-2] <= tiny_mistral_config.window_size

    def test_prefill_then_decode_loop(self, tiny_mistral, tiny_mistral_config):
        from snapmind.engine.decode import decode_step
        from snapmind.engine.prefill import prefill
        from snapmind.sampling.greedy import GreedySampler

        sampler = GreedySampler()
        kv_cache = {i: {"k": None, "v": None} for i in range(tiny_mistral_config.n_layers)}
        prompt = torch.randint(0, 32000, (1, 8))
        last_logits, _ = prefill(tiny_mistral, prompt, kv_cache)
        sampled = sampler.sample(last_logits[0, :])
        token_id = int(sampled.item()) if isinstance(sampled, torch.Tensor) else sampled

        step = 8
        for _ in range(3):
            token_id = decode_step(tiny_mistral, token_id, kv_cache, sampler, position_ids=torch.tensor([step]))
            step += 1
            for i in range(tiny_mistral_config.n_layers):
                k = kv_cache[i]["k"]
                assert k is not None
                assert k.shape[-2] <= tiny_mistral_config.window_size

    def test_prefill_sliding_window_mask(self, tiny_mistral):
        tokens = torch.randint(0, 32000, (1, 20))
        logits = tiny_mistral(tokens)
        assert logits.shape == (1, 20, 32000)

    def test_long_prompt_truncates_cache(self, tiny_mistral, tiny_mistral_config):
        kv_cache = {i: {"k": None, "v": None} for i in range(tiny_mistral_config.n_layers)}
        long_prompt = torch.randint(0, 32000, (1, 30))
        tiny_mistral(long_prompt, kv_cache=kv_cache)
        for i in range(tiny_mistral_config.n_layers):
            k = kv_cache[i]["k"]
            assert k is not None
            assert k.shape[-2] <= tiny_mistral_config.window_size
            assert k.shape[-2] <= 30


# ─── ENDSECTION: Mistral Integration Tests ───────────────
