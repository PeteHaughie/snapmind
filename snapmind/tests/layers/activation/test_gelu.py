import pytest
import torch


class TestGELUShape:
    def test_output_shape_matches_input(self, gelu, test_tensor):
        result = gelu(test_tensor)
        assert result.shape == (2, 16, 32)

    def test_batch_dimension_preserved(self, gelu):
        x = torch.randn(4, 16, 32)
        result = gelu(x)
        assert result.shape == (4, 16, 32)

    def test_arbitrary_shape(self, gelu):
        x = torch.randn(3, 7, 11, 13)
        result = gelu(x)
        assert result.shape == (3, 7, 11, 13)


class TestGELUValues:
    def test_zero_input_returns_zero(self, gelu):
        result = gelu(torch.tensor(0.0))
        assert torch.allclose(result, torch.tensor(0.0), atol=1e-6)

    def test_negative_one(self, gelu):
        result = gelu(torch.tensor(-1.0))
        expected = torch.tensor(-0.1588)
        assert torch.allclose(result, expected, atol=1e-3)

    def test_positive_one(self, gelu):
        result = gelu(torch.tensor(1.0))
        expected = torch.tensor(0.8413)
        assert torch.allclose(result, expected, atol=1e-3)

    def test_large_positive_is_approximately_identity(self, gelu):
        result = gelu(torch.tensor(100.0))
        assert torch.allclose(result, torch.tensor(100.0), atol=1e-3)

    def test_large_negative_is_approximately_zero(self, gelu):
        result = gelu(torch.tensor(-100.0))
        assert torch.allclose(result, torch.tensor(0.0), atol=1e-3)


class TestGELUProperties:
    def test_positive_monotonic(self, gelu):
        x = torch.linspace(0, 5, 50)
        result = gelu(x)
        diffs = result[1:] - result[:-1]
        assert (diffs >= 0).all()

    def test_negative_asymptotic_to_zero(self, gelu):
        x = torch.linspace(-10, -5, 50)
        result = gelu(x)
        assert torch.allclose(result, torch.zeros_like(result), atol=1e-2)
