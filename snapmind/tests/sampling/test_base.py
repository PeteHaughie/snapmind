import pytest
import torch


class TestSamplerABC:
    def test_cannot_instantiate_directly(self):
        from snapmind.sampling.base import SamplerABC
        with pytest.raises(TypeError):
            SamplerABC()

    def test_minimal_subclass_works(self):
        from snapmind.sampling.base import SamplerABC
        class MinimalSampler(SamplerABC):
            def sample(self, logits, temperature=1.0):
                return logits.argmax(dim=-1)

        sampler = MinimalSampler()
        logits = torch.randn(2, 16, 256)
        result = sampler.sample(logits)
        assert result.shape == (2, 16)

    def test_registered_via_SAMPLER(self):
        from snapmind.core.registry import SAMPLER
        assert "greedy" in SAMPLER
