import pytest
import torch


class TestPositionalEncodingABC:
    def test_cannot_instantiate_directly(self):
        from snapmind.layers.positional.base import PositionalEncodingABC
        with pytest.raises(TypeError):
            PositionalEncodingABC()

    def test_minimal_subclass_works(self):
        from snapmind.layers.positional.base import PositionalEncodingABC
        class MinimalPE(PositionalEncodingABC):
            @property
            def injection_point(self):
                return "embedding"

            def forward(self, x, position_ids=None):
                return x

        instance = MinimalPE()
        x = torch.randn(1, 4, 64)
        result = instance(x)
        assert result.shape == (1, 4, 64)

    def test_registered_via_PE(self):
        from snapmind.core.registry import PE
        assert "learned" in PE
        assert "none" in PE
