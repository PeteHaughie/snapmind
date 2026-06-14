import pytest
import torch


class TestNormABC:
    def test_cannot_instantiate_directly(self):
        from snapmind.layers.normalization.base import NormABC

        with pytest.raises(TypeError):
            NormABC()

    def test_minimal_subclass_works(self):
        from snapmind.layers.normalization.base import NormABC

        class MinimalNorm(NormABC):
            def forward(self, x):
                return x

        instance = MinimalNorm()
        x = torch.randn(1, 4, 64)
        result = instance(x)
        assert result.shape == (1, 4, 64)

    def test_registered_via_NORM(self):
        from snapmind.core.registry import NORM

        assert "layernorm" in NORM
