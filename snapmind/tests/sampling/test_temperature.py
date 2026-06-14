import pytest
import torch


class TestTemperatureSampler:
    def test_temperature_zero_is_argmax(self):
        from snapmind.sampling.temperature import TemperatureSampler
        sampler = TemperatureSampler()
        logits = torch.randn(100)
        result = sampler.sample(logits, temperature=0.0)
        assert result == logits.argmax().item()

    def test_temperature_one_passes_through(self):
        from snapmind.sampling.temperature import TemperatureSampler
        sampler = TemperatureSampler(wrapped="greedy")
        logits = torch.randn(100)
        result = sampler.sample(logits, temperature=1.0)
        assert result == logits.argmax().item()

    def test_temperature_high_is_stochastic(self):
        from snapmind.sampling.temperature import TemperatureSampler
        sampler = TemperatureSampler(wrapped="greedy")
        logits = torch.tensor([0.0, 0.001, 0.002, 0.003])
        results = [sampler.sample(logits, temperature=1000.0) for _ in range(200)]
        n_unique = len(set(results))
        assert n_unique > 1

    def test_initialized_from_registry(self):
        from snapmind.sampling.temperature import TemperatureSampler
        sampler = TemperatureSampler()
        logits = torch.randn(100)
        result = sampler.sample(logits, temperature=1.0)
        assert isinstance(result, (int, torch.Tensor))
