import pytest
import torch


class TestSiLUABC:
    def test_cannot_instantiate_directly(self):
        from snapmind.layers.activation.base import ActivationABC

        with pytest.raises(TypeError):
            ActivationABC()

    def test_minimal_subclass_works(self):
        from snapmind.layers.activation.base import ActivationABC

        class MinimalAct(ActivationABC):
            def forward(self, x):
                return x

        m = MinimalAct()
        t = torch.randn(2, 4)
        assert m(t).shape == (2, 4)

    def test_registered_in_activation_registry(self):
        from snapmind.core.registry import ACTIVATION

        assert "silu" in ACTIVATION


class TestSiLU:
    @pytest.fixture
    def silu(self):
        from snapmind.layers.activation.silu import SiLU

        return SiLU()

    def test_zero_input_gives_zero(self, silu):
        assert silu(torch.tensor(0.0)).item() == 0.0

    def test_large_positive_approx_identity(self, silu):
        x = torch.tensor(100.0)
        assert torch.isclose(silu(x), x, atol=1e-6)

    def test_large_negative_goes_to_zero(self, silu):
        x = torch.tensor(-100.0)
        assert silu(x).item() == pytest.approx(0.0, abs=1e-6)

    def test_negative_one_value(self, silu):
        x = torch.tensor(-1.0)
        expected = -1.0 * torch.sigmoid(torch.tensor(-1.0))
        assert torch.isclose(silu(x), expected, atol=1e-6)

    def test_is_monotonic_for_positive_inputs(self, silu):
        x = torch.linspace(0, 10, 200)
        y = silu(x)
        assert (y.diff() >= -1e-6).all()

    def test_output_shape_preserved(self, silu):
        x = torch.randn(2, 16, 32)
        out = silu(x)
        assert out.shape == x.shape
