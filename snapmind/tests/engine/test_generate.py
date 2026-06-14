import pytest


class TestGenerateEngine:
    def test_generate_yields_tokens(self, mini_gpt2):
        from snapmind.tokenizer.hf import HFTokenizer
        from snapmind.sampling.greedy import GreedySampler
        from snapmind.engine.generate import GenerateEngine
        import asyncio

        tok = HFTokenizer()
        engine = GenerateEngine(mini_gpt2, tok, GreedySampler())

        async def run():
            tokens = []
            async for token in engine.generate("Hello", max_tokens=5):
                tokens.append(token)
            return tokens

        tokens = asyncio.run(run())
        assert len(tokens) > 0
        assert all(isinstance(t, str) for t in tokens)

    def test_generate_stops_at_max_tokens(self, mini_gpt2):
        from snapmind.tokenizer.hf import HFTokenizer
        from snapmind.sampling.greedy import GreedySampler
        from snapmind.engine.generate import GenerateEngine
        import asyncio

        tok = HFTokenizer()
        engine = GenerateEngine(mini_gpt2, tok, GreedySampler(), eos_token_id=-1)

        async def run():
            tokens = []
            async for token in engine.generate("Hello", max_tokens=10):
                tokens.append(token)
            return tokens

        tokens = asyncio.run(run())
        assert len(tokens) == 10

    def test_generate_does_not_yield_empty_tokens(self, mini_gpt2):
        from snapmind.tokenizer.hf import HFTokenizer
        from snapmind.sampling.greedy import GreedySampler
        from snapmind.engine.generate import GenerateEngine
        import asyncio

        tok = HFTokenizer()
        engine = GenerateEngine(mini_gpt2, tok, GreedySampler())

        async def run():
            tokens = []
            async for token in engine.generate("Hello", max_tokens=10):
                assert len(token) > 0
                tokens.append(token)
            return tokens

        tokens = asyncio.run(run())
        assert len(tokens) > 0

    def test_generate_respects_eos_token(self, mini_gpt2, monkeypatch):
        from snapmind.tokenizer.hf import HFTokenizer
        from snapmind.sampling.greedy import GreedySampler
        from snapmind.engine.generate import GenerateEngine
        import asyncio
        import torch

        tok = HFTokenizer()
        engine = GenerateEngine(mini_gpt2, tok, GreedySampler(), eos_token_id=0)

        # Force model forward to make token 0 (EOS) the greedy choice
        def patched_forward(tokens, kv_cache=None):
            logits = torch.randn(tokens.shape[0], tokens.shape[1], 50257)
            logits[..., 0] = 100.0
            return logits

        monkeypatch.setattr(mini_gpt2, "forward", patched_forward)

        async def run():
            tokens = []
            async for token in engine.generate("Hello", max_tokens=100):
                tokens.append(token)
            return tokens

        tokens = asyncio.run(run())
        assert len(tokens) < 100
