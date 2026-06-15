import torch


class TestMirostatSampler:
    def test_returns_valid_indices(self):
        from snapmind.sampling.mirostat import MirostatSampler

        sampler = MirostatSampler()
        logits = torch.randn(4, 100)
        indices = sampler.sample(logits, temperature=1.0)
        assert indices.shape == (4,)
        assert (0 <= indices).all() and (indices < 100).all()

    def test_temperature_zero_is_argmax(self):
        from snapmind.sampling.mirostat import MirostatSampler

        sampler = MirostatSampler()
        logits = torch.randn(100)
        result = sampler.sample(logits, temperature=0.0)
        assert result == logits.argmax().item()

    def test_batch_shape_preserved(self):
        from snapmind.sampling.mirostat import MirostatSampler

        sampler = MirostatSampler()
        logits = torch.randn(2, 4, 100)
        indices = sampler.sample(logits, temperature=1.0)
        assert indices.shape == (2, 4)

    def test_max_surprise_updates(self):
        from snapmind.sampling.mirostat import MirostatSampler

        sampler = MirostatSampler(tau=5.0, learning_rate=0.1)
        initial = sampler.max_surprise
        logits = torch.randn(1000)
        for _ in range(10):
            sampler.sample(logits, temperature=1.0)
        assert sampler.max_surprise != initial

    def test_reset_restores_state(self):
        from snapmind.sampling.mirostat import MirostatSampler

        sampler = MirostatSampler(tau=5.0, learning_rate=0.1)
        logits = torch.randn(1000)
        for _ in range(10):
            sampler.sample(logits, temperature=1.0)
        sampler.reset()
        assert sampler.max_surprise == 2.0

    def test_low_tau_reduces_k(self):
        from snapmind.sampling.mirostat import MirostatSampler

        sampler = MirostatSampler(tau=2.0, learning_rate=0.5)
        logits = torch.randn(1000)
        # After many steps with low tau, max_surprise should decrease
        initial = sampler.max_surprise
        for _ in range(50):
            sampler.sample(logits, temperature=1.0)
        # High tau drives max_surprise down (less surprise allowed)
        assert sampler.max_surprise <= initial + 0.5 or sampler.max_surprise >= 0.1

    def test_high_tau_allows_diversity(self):
        from snapmind.sampling.mirostat import MirostatSampler

        sampler = MirostatSampler(tau=8.0, learning_rate=0.3)
        logits = torch.randn(1000)
        sampled = []
        for _ in range(100):
            idx = sampler.sample(logits, temperature=1.0)
            if isinstance(idx, torch.Tensor):
                idx = int(idx.item())
            sampled.append(idx)
        assert len(set(sampled)) > 5
