import torch


class TestGreedySampler:
    def test_returns_argmax(self):
        from snapmind.sampling.greedy import GreedySampler

        sampler = GreedySampler()
        logits = torch.tensor([[-10.0, -5.0, 0.0, 5.0, 10.0]])
        result = sampler.sample(logits)
        assert result.item() == 4

    def test_shape_matches_batch(self):
        from snapmind.sampling.greedy import GreedySampler

        sampler = GreedySampler()
        logits = torch.randn(4, 256)
        result = sampler.sample(logits)
        assert result.shape == (4,)

    def test_always_picks_highest(self):
        from snapmind.sampling.greedy import GreedySampler

        sampler = GreedySampler()
        for _ in range(100):
            logits = torch.randn(1, 100)
            result = sampler.sample(logits)
            assert result.item() == logits.argmax().item()

    def test_temperature_has_no_effect(self):
        from snapmind.sampling.greedy import GreedySampler

        sampler = GreedySampler()
        logits = torch.randn(1, 100)
        result_default = sampler.sample(logits)
        result_high = sampler.sample(logits, temperature=100.0)
        result_low = sampler.sample(logits, temperature=0.01)
        assert result_default.item() == result_high.item()
        assert result_default.item() == result_low.item()
