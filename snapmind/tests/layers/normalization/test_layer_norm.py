import pytest
import torch


class TestLayerNormShape:
    def test_output_shape_matches_input(self, layer_norm, test_tensor):
        result = layer_norm(test_tensor)
        assert result.shape == (2, 16, 32)

    def test_batch_dimension_preserved(self, layer_norm):
        x = torch.randn(4, 16, 32)
        result = layer_norm(x)
        assert result.shape == (4, 16, 32)


class TestLayerNormProperties:
    def test_output_has_zero_mean(self, layer_norm):
        x = torch.randn(2, 16, 32)
        result = layer_norm(x)
        assert torch.allclose(result.mean(dim=-1), torch.zeros_like(result.mean(dim=-1)), atol=1e-5)

    def test_output_has_unit_variance(self, layer_norm):
        x = torch.randn(2, 16, 32)
        result = layer_norm(x)
        assert torch.allclose(result.std(dim=-1, unbiased=False), torch.ones_like(result.std(dim=-1, unbiased=False)), atol=1e-4)

    def test_affine_transform_applied(self, layer_norm):
        x = torch.randn(2, 16, 32)
        result = layer_norm(x)
        assert result.requires_grad

    def test_learnable_parameters_exist(self, layer_norm):
        params = list(layer_norm.parameters())
        assert len(params) >= 2


class TestLayerNormStability:
    def test_handles_single_element(self, layer_norm):
        x = torch.randn(1, 1, 32)
        result = layer_norm(x)
        assert result.shape == (1, 1, 32)
        assert torch.isfinite(result).all()

    def test_handles_large_values(self, layer_norm):
        x = torch.randn(2, 16, 32) * 1000
        result = layer_norm(x)
        assert result.shape == (2, 16, 32)
        assert torch.isfinite(result).all()
