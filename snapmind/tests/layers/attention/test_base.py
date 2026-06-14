import pytest
import torch


class TestAttentionABC:
    def test_cannot_instantiate_directly(self):
        from snapmind.layers.attention.base import AttentionABC
        with pytest.raises(TypeError):
            AttentionABC()

    def test_minimal_subclass_works(self):
        from snapmind.layers.attention.base import AttentionABC
        class MinimalAttention(AttentionABC):
            def forward(self, x, kv_cache=None, position_ids=None, mask=None):
                return x
        instance = MinimalAttention()
        x = torch.randn(1, 4, 64)
        result = instance(x)
        assert result.shape == (1, 4, 64)

    def test_all_abstract_methods_must_be_implemented(self):
        from snapmind.layers.attention.base import AttentionABC
        with pytest.raises(TypeError):

            class PartialAttention(AttentionABC):
                pass

            PartialAttention()

    def test_registered_via_ATTENTION(self):
        from snapmind.core.registry import ATTENTION
        assert "sdpa" in ATTENTION
