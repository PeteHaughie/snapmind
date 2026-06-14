import torch


class TestLearnedPositionalEncodingShape:
    def test_output_shape_matches_input(self, learned_pe):
        x = torch.randn(2, 16, 32)
        result = learned_pe(x)
        assert result.shape == (2, 16, 32)

    def test_batch_dimension_preserved(self, learned_pe):
        x = torch.randn(4, 16, 32)
        result = learned_pe(x)
        assert result.shape == (4, 16, 32)


class TestLearnedPositionalEncodingProperties:
    def test_different_positions_have_different_encodings(self, learned_pe):
        x = torch.zeros(1, 4, 32)
        output = learned_pe(x)
        assert not torch.allclose(output[0, 0], output[0, 1], atol=1e-6)
        assert not torch.allclose(output[0, 0], output[0, 2], atol=1e-6)
        assert not torch.allclose(output[0, 1], output[0, 2], atol=1e-6)

    def test_same_position_across_batch_is_identical(self, learned_pe):
        x = torch.zeros(3, 10, 32)
        output = learned_pe(x)
        assert torch.allclose(output[0, 0], output[1, 0], atol=1e-6)
        assert torch.allclose(output[0, 0], output[2, 0], atol=1e-6)
        assert torch.allclose(output[1, 3], output[2, 3], atol=1e-6)

    def test_zero_input_returns_position_encoding_only(self, learned_pe):
        x = torch.zeros(1, 10, 32)
        output = learned_pe(x)
        assert not torch.allclose(output, torch.zeros_like(output))
