import torch


class TestGPT2Architecture:
    def test_parameter_count_124m(self, gpt2_124m_config):
        from snapmind.models.gpt2 import GPT2Model

        model = GPT2Model(gpt2_124m_config)
        n_params = sum(p.numel() for p in model.parameters())
        assert abs(n_params - 124_000_000) < 10_000_000

    def test_weight_tying(self, gpt2_124m_config):
        from snapmind.models.gpt2 import GPT2Model

        model = GPT2Model(gpt2_124m_config)
        assert model.lm_head.weight is model.embed.weight

    def test_embedding_dim_matches_vocab(self, gpt2_124m_config):
        from snapmind.models.gpt2 import GPT2Model

        model = GPT2Model(gpt2_124m_config)
        assert model.embed.weight.shape == (50257, 768)
        assert model.lm_head.weight.shape == (50257, 768)

    def test_correct_number_of_layers(self, gpt2_124m_config):
        from snapmind.models.gpt2 import GPT2Model

        model = GPT2Model(gpt2_124m_config)
        assert len(model.layers) == 12


class TestGPT2Forward:
    def test_output_logits_shape(self, tiny_gpt2, test_tokens):
        logits = tiny_gpt2(test_tokens)
        batch, seq_len = test_tokens.shape
        assert logits.shape == (batch, seq_len, tiny_gpt2.config.vocab_size)

    def test_all_outputs_finite(self, tiny_gpt2, test_tokens):
        logits = tiny_gpt2(test_tokens)
        assert torch.isfinite(logits).all()

    def test_batch_independent(self, tiny_gpt2):
        tokens = torch.randint(0, 256, (4, 16))
        logits = tiny_gpt2(tokens)
        assert logits.shape[0] == 4

    def test_logits_not_all_identical(self, tiny_gpt2, test_tokens):
        logits = tiny_gpt2(test_tokens)
        assert not torch.allclose(logits[0, 0], logits[0, 1], atol=1e-4)


class TestGPT2CausalMask:
    def test_random_weights_do_not_produce_identical_logits(self, tiny_gpt2, test_tokens):
        logits = tiny_gpt2(test_tokens)
        assert logits.shape[-1] == tiny_gpt2.config.vocab_size
        assert not torch.allclose(logits[:, :1, :], logits[:, -1:, :], atol=1e-2)
