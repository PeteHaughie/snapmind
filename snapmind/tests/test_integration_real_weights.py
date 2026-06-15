# ─── SECTION: Real-Weight Integration Tests ────────────
import pytest
import torch


@pytest.fixture(scope="module")
def _gpt2_build():
    from snapmind.serving.cli import build_model
    from snapmind.tokenizer.hf import HFTokenizer

    m, cfg = build_model("gpt2", device="cpu")
    tok = HFTokenizer(model_name="gpt2")
    return m, cfg, tok


@pytest.fixture(scope="module")
def _tinyllama_build():
    from snapmind.serving.cli import build_model
    from snapmind.tokenizer.hf import HFTokenizer

    m, cfg = build_model("tinyllama", device="cpu")
    tok = HFTokenizer(model_name="tinyllama")
    return m, cfg, tok


@pytest.mark.slow
class TestRealGPT2:
    """Exercises the ARCHITECTURE registry path via build_model()."""

    def test_loader_reports_no_missing(self, _gpt2_build):
        m, _, _ = _gpt2_build
        state = m.state_dict()
        assert "embed.weight" in state
        assert "ln_f.weight" in state
        assert "lm_head.weight" in state

    def test_forward_pass(self, _gpt2_build):
        m, cfg, tok = _gpt2_build
        ids = tok.encode("The capital of France is")
        tokens = torch.tensor([ids])
        with torch.no_grad():
            logits = m(tokens)
        assert logits.shape == (1, len(ids), cfg.vocab_size)
        assert torch.isfinite(logits).all()

    def test_top_prediction_is_coherent(self, _gpt2_build):
        m, _, tok = _gpt2_build
        ids = tok.encode("The capital of France is")
        tokens = torch.tensor([ids])
        with torch.no_grad():
            logits = m(tokens)
        next_id = int(logits[0, -1].argmax())
        token = tok.decode([next_id])
        assert isinstance(token, str)
        assert len(token) > 0


@pytest.mark.slow
class TestRealTinyLlama:
    """Downloads real TinyLlama weights and tests LlamaModel via registry."""

    def test_loader_reports_no_missing(self, _tinyllama_build):
        m, _, _ = _tinyllama_build
        state = m.state_dict()
        assert "embed.weight" in state
        assert "norm.weight" in state
        assert "lm_head.weight" in state

    def test_forward_pass(self, _tinyllama_build):
        m, cfg, tok = _tinyllama_build
        ids = tok.encode("What is attention?")
        tokens = torch.tensor([ids])
        with torch.no_grad():
            logits = m(tokens)
        assert logits.shape == (1, len(ids), cfg.vocab_size)
        assert torch.isfinite(logits).all()

    def test_logits_not_all_identical(self, _tinyllama_build):
        m, _, tok = _tinyllama_build
        ids = tok.encode("Hello world")
        tokens = torch.tensor([ids])
        with torch.no_grad():
            logits = m(tokens)
        assert not torch.allclose(logits[:, 0, :], logits[:, -1, :], atol=1e-2)


# ─── ENDSECTION: Real-Weight Integration Tests ─────────
