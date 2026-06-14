import pytest
import torch
import torch.nn.functional as F


class TestTopPSampler:
    def test_top_p_returns_valid_indices(self):
        from snapmind.sampling.top_p import TopPSampler
        sampler = TopPSampler()
        logits = torch.randn(4, 100)
        indices = sampler.sample(logits, temperature=1.0, top_p=0.9)
        assert indices.shape == (4,)
        assert (0 <= indices).all() and (indices < 100).all()

    def test_top_p_1_is_essentially_full_distribution(self):
        from snapmind.sampling.top_p import TopPSampler
        sampler = TopPSampler()
        logits = torch.randn(1000)
        indices = torch.tensor([sampler.sample(logits, temperature=1.0, top_p=1.0) for _ in range(500)])
        n_unique = indices.unique().numel()
        assert n_unique > 10

    def test_top_p_barely_any_tokens(self):
        from snapmind.sampling.top_p import TopPSampler
        sampler = TopPSampler()
        logits = torch.tensor([-100.0, -100.0, 0.0, -100.0, -100.0])
        indices = torch.tensor([sampler.sample(logits, temperature=1.0, top_p=0.5) for _ in range(100)])
        assert (indices == 2).all()

    def test_temperature_zero_is_argmax(self):
        from snapmind.sampling.top_p import TopPSampler
        sampler = TopPSampler()
        logits = torch.randn(100)
        result = sampler.sample(logits, temperature=0.0, top_p=0.9)
        assert result == logits.argmax().item()

    def test_batch_shape_preserved(self):
        from snapmind.sampling.top_p import TopPSampler
        sampler = TopPSampler()
        logits = torch.randn(2, 4, 100)
        indices = sampler.sample(logits, temperature=1.0, top_p=0.9)
        assert indices.shape == (2, 4)
