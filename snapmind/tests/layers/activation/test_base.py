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

    def test_activations_exist(self):
        from snapmind.layers.activation.base import ActivationABC
        from snapmind.layers.activation.gelu import GELU
        from snapmind.layers.activation.silu import SiLU

        assert issubclass(GELU, ActivationABC)
        assert issubclass(SiLU, ActivationABC)
