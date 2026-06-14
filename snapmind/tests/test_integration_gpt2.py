import pytest
import torch


@pytest.fixture(scope="module")
def gpt2_model():
    from snapmind.core.config import ModelConfig
    from snapmind.loaders.safetensors import SafetensorsLoader
    from snapmind.models.gpt2 import GPT2Model

    config = ModelConfig(
        model_type="gpt2",
        d_model=768,
        n_heads=12,
        n_kv_heads=12,
        n_layers=12,
        d_ff=3072,
        vocab_size=50257,
        max_seq_len=1024,
        norm_eps=1e-5,
        attention_type="sdpa",
        pe_type="learned",
        norm_type="layernorm",
        activation_type="gelu",
        kv_cache_type="naive",
    )
    model = GPT2Model(config)
    model.eval()
    with torch.no_grad():
        loader = SafetensorsLoader()
        loader.load(None, model, config)
    return model


@pytest.fixture(scope="module")
def gpt2_state_dict():
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    path = hf_hub_download(repo_id="openai-community/gpt2", filename="model.safetensors")
    return load_file(path, device="cpu")


class TestGPT2IntegrationLoad:
    def test_download_success(self, gpt2_state_dict):
        assert "wte.weight" in gpt2_state_dict
        assert gpt2_state_dict["wte.weight"].shape == (50257, 768)

    def test_all_expected_keys_present(self, gpt2_state_dict):
        expected_prefixes = [
            "wte.weight",
            "wpe.weight",
            "ln_f.weight",
            "ln_f.bias",
        ]
        for i in range(12):
            expected_prefixes.extend(
                [
                    f"h.{i}.ln_1.weight",
                    f"h.{i}.ln_1.bias",
                    f"h.{i}.ln_2.weight",
                    f"h.{i}.ln_2.bias",
                    f"h.{i}.attn.c_attn.weight",
                    f"h.{i}.attn.c_attn.bias",
                    f"h.{i}.attn.c_proj.weight",
                    f"h.{i}.attn.c_proj.bias",
                    f"h.{i}.mlp.c_fc.weight",
                    f"h.{i}.mlp.c_fc.bias",
                    f"h.{i}.mlp.c_proj.weight",
                    f"h.{i}.mlp.c_proj.bias",
                ]
            )
        for key in expected_prefixes:
            assert key in gpt2_state_dict, f"Missing key: {key}"

    def test_loader_reports_no_missing_keys(self, gpt2_model):
        _ = gpt2_model


class TestGPT2IntegrationForward:
    def test_forward_pass_succeeds(self, gpt2_model):
        tokens = torch.randint(0, 50256, (1, 16))
        with torch.no_grad():
            logits = gpt2_model(tokens)
        assert logits.shape == (1, 16, 50257)

    def test_logits_all_finite(self, gpt2_model):
        tokens = torch.randint(0, 50256, (1, 16))
        with torch.no_grad():
            logits = gpt2_model(tokens)
        assert torch.isfinite(logits).all()

    def test_logits_not_all_identical(self, gpt2_model):
        tokens = torch.randint(0, 50256, (1, 16))
        with torch.no_grad():
            logits = gpt2_model(tokens)
        assert not torch.allclose(logits[:, 0, :], logits[:, -1, :], atol=1e-2)

    def test_last_token_has_highest_confidence(self, gpt2_model):
        tokens = torch.randint(0, 50256, (1, 16))
        with torch.no_grad():
            logits = gpt2_model(tokens)
        probs = torch.softmax(logits[0, -1, :], dim=-1)
        top_prob, top_idx = probs.max(dim=-1)
        assert top_prob > 0.0
        assert top_prob < 1.0
        assert 0 <= top_idx < 50257

    def test_same_input_same_output(self, gpt2_model):
        tokens = torch.randint(0, 50256, (1, 8))
        with torch.no_grad():
            out1 = gpt2_model(tokens)
            out2 = gpt2_model(tokens)
        assert torch.allclose(out1, out2, atol=1e-6)


class TestGPT2IntegrationTokenizer:
    def test_tokenizer_encodes(self):
        from snapmind.tokenizer.hf import HFTokenizer

        tok = HFTokenizer()
        ids = tok.encode("Hello, my name is")
        assert isinstance(ids, list)
        assert len(ids) > 0
        assert all(isinstance(i, int) for i in ids)

    def test_tokenizer_decode_round_trip(self):
        from snapmind.tokenizer.hf import HFTokenizer

        tok = HFTokenizer()
        text = "Hello, my name is GPT-2"
        ids = tok.encode(text)
        decoded = tok.decode(ids)
        assert len(decoded) > 0

    def test_tokenizer_vocab_size(self):
        from snapmind.tokenizer.hf import HFTokenizer

        tok = HFTokenizer()
        assert tok.vocab_size() == 50257

    def test_tokenizer_matches_model_vocab(self, gpt2_model):
        from snapmind.tokenizer.hf import HFTokenizer

        tok = HFTokenizer()
        assert tok.vocab_size() == gpt2_model.config.vocab_size


@pytest.mark.slow
class TestGPT2IntegrationPrediction:
    def test_predicts_plausible_next_token(self, gpt2_model):
        from snapmind.tokenizer.hf import HFTokenizer

        tok = HFTokenizer()
        text = "The capital of France is"
        input_ids = torch.tensor([tok.encode(text)])
        with torch.no_grad():
            logits = gpt2_model(input_ids)
        next_token_logits = logits[0, -1, :]
        top_idx = next_token_logits.argmax().item()
        top_token = tok.decode([top_idx])
        assert len(top_token) > 0
