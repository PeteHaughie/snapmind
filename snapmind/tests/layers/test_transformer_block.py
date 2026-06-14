import torch


class TestTransformerBlockShape:
    def test_output_shape_matches_input(self, tiny_config, test_tokens):
        from snapmind.layers.transformer_block import TransformerBlock

        block = TransformerBlock(
            config=tiny_config,
            layer_idx=0,
        )
        x = torch.randn(2, 16, tiny_config.d_model)
        result = block(x, kv_cache=None, position_ids=None)
        assert result.shape == (2, 16, tiny_config.d_model)

    def test_batch_dimension_preserved(self, tiny_config):
        from snapmind.layers.transformer_block import TransformerBlock

        block = TransformerBlock(config=tiny_config, layer_idx=0)
        x = torch.randn(4, 16, tiny_config.d_model)
        result = block(x)
        assert result.shape == (4, 16, tiny_config.d_model)


class TestTransformerBlockProperties:
    def test_residual_is_preserved(self, tiny_config):
        from snapmind.layers.transformer_block import TransformerBlock

        block = TransformerBlock(config=tiny_config, layer_idx=0)
        x = torch.randn(2, 16, tiny_config.d_model)
        result = block(x)
        assert not torch.allclose(result, torch.zeros_like(result))

    def test_residual_adds_positive_contribution(self, tiny_config):
        from snapmind.layers.transformer_block import TransformerBlock

        block = TransformerBlock(config=tiny_config, layer_idx=0)
        x = torch.randn(2, 16, tiny_config.d_model)
        result = block(x)
        assert not torch.allclose(result, x, atol=1e-2)

    def test_all_outputs_finite(self, tiny_config):
        from snapmind.layers.transformer_block import TransformerBlock

        block = TransformerBlock(config=tiny_config, layer_idx=0)
        x = torch.randn(2, 16, tiny_config.d_model)
        result = block(x)
        assert torch.isfinite(result).all()


class TestTransformerBlockGradients:
    def test_gradients_flow_through_block(self, tiny_config):
        from snapmind.layers.transformer_block import TransformerBlock

        block = TransformerBlock(config=tiny_config, layer_idx=0)
        x = torch.randn(2, 16, tiny_config.d_model, requires_grad=True)
        result = block(x)
        loss = result.sum()
        loss.backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
