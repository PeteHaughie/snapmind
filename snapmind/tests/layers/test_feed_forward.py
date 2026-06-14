import torch


class TestFeedForwardShape:
    def test_output_shape_matches_input(self, tiny_config):
        from snapmind.layers.feed_forward import FeedForward

        ffn = FeedForward(
            d_model=tiny_config.d_model,
            d_ff=tiny_config.d_ff,
            activation_type=tiny_config.activation_type,
            dropout=0.0,
        )
        x = torch.randn(2, 16, tiny_config.d_model)
        result = ffn(x)
        assert result.shape == (2, 16, tiny_config.d_model)

    def test_batch_dimension_preserved(self, tiny_config):
        from snapmind.layers.feed_forward import FeedForward

        ffn = FeedForward(
            d_model=tiny_config.d_model,
            d_ff=tiny_config.d_ff,
            activation_type=tiny_config.activation_type,
            dropout=0.0,
        )
        x = torch.randn(4, 16, tiny_config.d_model)
        result = ffn(x)
        assert result.shape == (4, 16, tiny_config.d_model)


class TestFeedForwardValues:
    def test_biased_output(self, tiny_config):
        from snapmind.layers.feed_forward import FeedForward

        ffn = FeedForward(
            d_model=tiny_config.d_model,
            d_ff=tiny_config.d_ff,
            activation_type=tiny_config.activation_type,
            dropout=0.0,
        )
        x = torch.zeros(2, 16, tiny_config.d_model)
        result = ffn(x)
        assert not torch.allclose(result, torch.zeros_like(result), atol=1e-6)
        assert result.shape == (2, 16, tiny_config.d_model)
        assert torch.isfinite(result).all()

    def test_all_outputs_finite(self, tiny_config):
        from snapmind.layers.feed_forward import FeedForward

        ffn = FeedForward(
            d_model=tiny_config.d_model,
            d_ff=tiny_config.d_ff,
            activation_type=tiny_config.activation_type,
            dropout=0.0,
        )
        x = torch.randn(2, 16, tiny_config.d_model)
        result = ffn(x)
        assert torch.isfinite(result).all()


class TestFeedForwardGradients:
    def test_gradients_flow(self, tiny_config):
        from snapmind.layers.feed_forward import FeedForward

        ffn = FeedForward(
            d_model=tiny_config.d_model,
            d_ff=tiny_config.d_ff,
            activation_type=tiny_config.activation_type,
            dropout=0.0,
        )
        x = torch.randn(2, 16, tiny_config.d_model, requires_grad=True)
        result = ffn(x)
        loss = result.sum()
        loss.backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
