import torch


class TestLlamaModel:
    def test_no_weight_tying(self, tiny_llama):
        assert tiny_llama.lm_head.weight is not tiny_llama.embed.weight

    def test_output_logits_shape(self, tiny_llama):
        tokens = torch.randint(0, 32000, (2, 8))
        logits = tiny_llama(tokens)
        assert logits.shape == (2, 8, 32000)

    def test_logits_are_finite(self, tiny_llama):
        tokens = torch.randint(0, 32000, (2, 8))
        logits = tiny_llama(tokens)
        assert torch.isfinite(logits).all()

    def test_embedding_weight_shape(self, tiny_llama):
        assert tiny_llama.embed.weight.shape == (32000, 64)

    def test_lm_head_weight_shape(self, tiny_llama):
        assert tiny_llama.lm_head.weight.shape == (32000, 64)

    def test_has_correct_number_of_layers(self, tiny_llama):
        assert len(tiny_llama.layers) == 2

    def test_uses_rms_norm(self, tiny_llama):
        from snapmind.layers.normalization.rms_norm import RMSNorm

        assert isinstance(tiny_llama.norm, RMSNorm)
        for layer in tiny_llama.layers:
            assert isinstance(layer.input_layernorm, RMSNorm)
            assert isinstance(layer.post_attention_layernorm, RMSNorm)

    def test_uses_gqa(self, tiny_llama):
        from snapmind.layers.attention.gqa import GroupedQueryAttention

        assert isinstance(tiny_llama.layers[0].self_attn, GroupedQueryAttention)

    def test_uses_gated_ffn(self, tiny_llama):
        from snapmind.layers.gated_feed_forward import GatedFeedForward

        assert isinstance(tiny_llama.layers[0].mlp, GatedFeedForward)

    def test_uses_rope(self, tiny_llama):
        from snapmind.layers.positional.rope import RotaryPositionalEncoding

        assert isinstance(tiny_llama.pe, RotaryPositionalEncoding)

    def test_registered_in_architecture_registry(self):
        from snapmind.core.architecture import ARCHITECTURE

        assert "llama" in ARCHITECTURE
        assert "tinyllama" in ARCHITECTURE

    def test_forward_with_kv_cache(self, tiny_llama):
        cfg = tiny_llama.config
        tokens = torch.randint(0, 32000, (1, 8))
        kv_cache = {i: {"k": None, "v": None} for i in range(cfg.n_layers)}
        logits = tiny_llama(tokens, kv_cache=kv_cache)
        assert logits.shape == (1, 8, 32000)
        for i in range(cfg.n_layers):
            assert kv_cache[i]["k"] is not None
            assert kv_cache[i]["v"] is not None

    def test_deterministic_output(self, tiny_llama):
        tokens = torch.randint(0, 32000, (1, 8))
        logits1 = tiny_llama(tokens)
        logits2 = tiny_llama(tokens)
        assert torch.equal(logits1, logits2)
