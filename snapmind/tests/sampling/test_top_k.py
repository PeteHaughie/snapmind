import torch


class TestTopKSampler:
    def test_top_k_returns_valid_indices(self):
        from snapmind.sampling.top_k import TopKSampler

        sampler = TopKSampler()
        logits = torch.randn(4, 100)
        indices = sampler.sample(logits, temperature=1.0, top_k=20)
        assert indices.shape == (4,)
        assert (0 <= indices).all() and (indices < 100).all()

    def test_top_k_1_is_argmax(self):
        from snapmind.sampling.top_k import TopKSampler

        sampler = TopKSampler()
        logits = torch.randn(100)
        indices = torch.tensor([sampler.sample(logits, temperature=1.0, top_k=1) for _ in range(100)])
        argmax = logits.argmax()
        assert (indices == argmax).all()

    def test_top_k_equal_vocab_is_full_distribution(self):
        from snapmind.sampling.top_k import TopKSampler

        sampler = TopKSampler()
        logits = torch.randn(1000)
        indices = torch.tensor([sampler.sample(logits, temperature=1.0, top_k=1000) for _ in range(500)])
        assert indices.unique().numel() > 10

    def test_temperature_zero_is_argmax(self):
        from snapmind.sampling.top_k import TopKSampler

        sampler = TopKSampler()
        logits = torch.randn(100)
        result = sampler.sample(logits, temperature=0.0, top_k=50)
        assert result == logits.argmax().item()

    def test_batch_shape_preserved(self):
        from snapmind.sampling.top_k import TopKSampler

        sampler = TopKSampler()
        logits = torch.randn(2, 4, 100)
        indices = sampler.sample(logits, temperature=1.0, top_k=20)
        assert indices.shape == (2, 4)

    def test_top_k_0_is_full_distribution(self):
        from snapmind.sampling.top_k import TopKSampler

        sampler = TopKSampler()
        logits = torch.randn(1000)
        indices = torch.tensor([sampler.sample(logits, temperature=1.0, top_k=0) for _ in range(500)])
        assert indices.unique().numel() > 10
