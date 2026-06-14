import pytest
import torch


class TestGatedFeedForward:
    @pytest.fixture
    def gated_ffn(self):
        from snapmind.layers.gated_feed_forward import GatedFeedForward
        return GatedFeedForward(d_model=32, d_ff=128)

    def test_output_shape(self, gated_ffn):
        x = torch.randn(2, 8, 32)
        out = gated_ffn(x)
        assert out.shape == (2, 8, 32)

    def test_has_three_projections(self, gated_ffn):
        assert hasattr(gated_ffn, "gate_proj")
        assert hasattr(gated_ffn, "up_proj")
        assert hasattr(gated_ffn, "down_proj")

    def test_projection_dimensions(self, gated_ffn):
        assert gated_ffn.gate_proj.weight.shape == (128, 32)
        assert gated_ffn.up_proj.weight.shape == (128, 32)
        assert gated_ffn.down_proj.weight.shape == (32, 128)

    def test_no_bias(self, gated_ffn):
        assert gated_ffn.gate_proj.bias is None
        assert gated_ffn.up_proj.bias is None
        assert gated_ffn.down_proj.bias is None

    def test_forward_is_differentiable(self, gated_ffn):
        x = torch.randn(2, 8, 32, requires_grad=True)
        out = gated_ffn(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
