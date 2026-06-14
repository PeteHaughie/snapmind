import pytest
import torch


class TestRMSNormABC:
    def test_cannot_instantiate_directly(self):
        from snapmind.layers.normalization.base import NormABC
        with pytest.raises(TypeError):
            NormABC()

    def test_minimal_subclass_works(self):
        from snapmind.layers.normalization.base import NormABC
        import torch.nn as nn
        class MinimalNorm(NormABC):
            def forward(self, x):
                return x
        m = MinimalNorm()
        t = torch.randn(2, 4)
        assert m(t).shape == (2, 4)

    def test_registered_in_norm_registry(self):
        from snapmind.core.registry import NORM
        assert "rmsnorm" in NORM


class TestRMSNorm:
    @pytest.fixture
    def rms_norm(self):
        from snapmind.layers.normalization.rms_norm import RMSNorm
        return RMSNorm(normalized_shape=32)

    def test_output_variance_approx_one(self, rms_norm):
        x = torch.randn(2, 16, 32)
        out = rms_norm(x)
        assert torch.allclose(out.std(dim=-1), torch.ones(2, 16), atol=0.15)

    def test_mean_not_forced_to_zero(self, rms_norm):
        x = torch.ones(2, 16, 32) * 5.0
        out = rms_norm(x)
        assert out.mean() > 0.01

    def test_affine_shift_preserves_shape(self, rms_norm):
        x = torch.randn(2, 16, 32)
        out = rms_norm(x)
        assert out.shape == x.shape

    def test_weight_learnable_parameter(self, rms_norm):
        assert hasattr(rms_norm, "weight")
        assert rms_norm.weight.shape == (32,)

    def test_different_eps_affects_output(self):
        from snapmind.layers.normalization.rms_norm import RMSNorm
        x = torch.randn(2, 16, 32) * 0.001
        rn1 = RMSNorm(normalized_shape=32, eps=1.0)
        rn2 = RMSNorm(normalized_shape=32, eps=1e-5)
        assert not torch.allclose(rn1(x), rn2(x), atol=1e-4)
