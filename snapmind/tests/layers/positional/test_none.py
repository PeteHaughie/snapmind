# ─── SECTION: NoPositionalEncoding Tests ─────────────────
import torch

from snapmind.layers.positional.none import NoPositionalEncoding


class TestNoPositionalEncodingContract:
    def test_is_pass_through(self):
        encoding = NoPositionalEncoding()
        x = torch.randn(2, 4, 64)
        result = encoding(x)
        assert torch.equal(result, x)

    def test_injection_point(self):
        encoding = NoPositionalEncoding()
        assert encoding.injection_point == "embedding"

    def test_preserves_dtype_and_device(self):
        encoding = NoPositionalEncoding()
        x = torch.randn(2, 4, 64, dtype=torch.float64)
        result = encoding(x)
        assert result.dtype == torch.float64

    def test_accepts_position_ids(self):
        encoding = NoPositionalEncoding()
        x = torch.randn(2, 4, 64)
        pos = torch.tensor([[0, 1, 2, 3]])
        result = encoding(x, position_ids=pos)
        assert torch.equal(result, x)

    def test_different_batch_sizes(self):
        encoding = NoPositionalEncoding()
        for b in [1, 2, 8]:
            x = torch.randn(b, 4, 64)
            result = encoding(x)
            assert result.shape == x.shape

    def test_apply_to_qk_is_noop(self):
        encoding = NoPositionalEncoding()
        q = torch.randn(2, 8, 4, 64)
        k = torch.randn(2, 8, 4, 64)
        q_out, k_out = encoding.apply_to_qk(q, k)
        assert torch.equal(q_out, q)
        assert torch.equal(k_out, k)


# ─── ENDSECTION: NoPositionalEncoding Tests ──────────────
