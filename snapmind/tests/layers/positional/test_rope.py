import pytest
import torch


class TestRoPEABC:
    def test_cannot_instantiate_directly(self):
        from snapmind.layers.positional.base import PositionalEncodingABC

        with pytest.raises(TypeError):
            PositionalEncodingABC()

    def test_minimal_subclass_works(self):
        from snapmind.layers.positional.base import PositionalEncodingABC

        class MinimalPE(PositionalEncodingABC):
            @property
            def injection_point(self):
                return "embedding"

            def forward(self, x, position_ids=None):
                return x

        m = MinimalPE()
        t = torch.randn(2, 4, 8)
        assert m(t).shape == (2, 4, 8)

    def test_registered_in_pe_registry(self):
        from snapmind.core.registry import PE

        assert "rope" in PE


class TestRotaryPositionalEncoding:
    @pytest.fixture
    def rope(self):
        from snapmind.layers.positional.rope import RotaryPositionalEncoding

        return RotaryPositionalEncoding(dim=8, max_seq_len=64)

    def test_forward_is_identity(self, rope):
        x = torch.randn(2, 8, 8)
        out = rope(x)
        assert torch.equal(out, x)

    def test_apply_to_qk_preserves_norm(self, rope):
        q = torch.randn(2, 4, 8, 8)
        k = torch.randn(2, 4, 8, 8)
        q_rot, k_rot = rope.apply_to_qk(q, k)
        assert torch.allclose(q.norm(dim=-1), q_rot.norm(dim=-1), atol=1e-5)
        assert torch.allclose(k.norm(dim=-1), k_rot.norm(dim=-1), atol=1e-5)

    def test_apply_to_qk_shape_preserved(self, rope):
        q = torch.randn(2, 4, 8, 8)
        k = torch.randn(2, 4, 8, 8)
        q_rot, k_rot = rope.apply_to_qk(q, k)
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

    def test_different_positions_produce_different_rotations(self, rope):
        q = torch.ones(1, 1, 4, 8)
        k = torch.ones(1, 1, 4, 8)
        pos0 = torch.arange(4).unsqueeze(0)
        pos1 = (torch.arange(4) + 10).unsqueeze(0)
        _, k_pos0 = rope.apply_to_qk(q, k, pos0)
        _, k_pos1 = rope.apply_to_qk(q, k, pos1)
        assert not torch.allclose(k_pos0, k_pos1)

    def test_injection_point_is_attention(self, rope):
        assert rope.injection_point == "attention"

    def test_rotate_half_swaps_sign(self, rope):
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        rotated = rope._rotate_half(x)
        expected = torch.tensor([[-3.0, -4.0, 1.0, 2.0]])
        assert torch.equal(rotated, expected)

    def test_apply_to_qk_first_token_no_rotation(self, rope):
        q = torch.tensor([[[[1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]]]])
        q_rot, _ = rope.apply_to_qk(q, q)
        assert torch.allclose(q_rot, q, atol=1e-6)
