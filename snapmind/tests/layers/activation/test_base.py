import pytest
import torch


class TestActivationABC:
    def test_cannot_instantiate_directly(self):
        from snapmind.layers.activation.base import ActivationABC

        with pytest.raises(TypeError):
            ActivationABC()

    def test_minimal_subclass_works(self):
        from snapmind.layers.activation.base import ActivationABC

        class MinimalActivation(ActivationABC):
            def forward(self, x):
                return x

        instance = MinimalActivation()
        x = torch.randn(2, 16, 32)
        result = instance(x)
        assert result.shape == (2, 16, 32)

    def test_registered_via_ACTIVATION(self):
        from snapmind.core.registry import ACTIVATION

        assert "gelu" in ACTIVATION
