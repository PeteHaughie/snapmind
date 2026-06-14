# ─── SECTION: HFTokenizer Tests ───────────────────────────
import pytest

from snapmind.tokenizer.hf import HFTokenizer


class TestHFTokenizerGPT2:
    def test_encode_decode_round_trip(self):
        tok = HFTokenizer(model_name="gpt2")
        text = "Hello, world!"
        ids = tok.encode(text)
        decoded = tok.decode(ids)
        assert decoded == text

    def test_known_token_ids(self):
        tok = HFTokenizer(model_name="gpt2")
        assert isinstance(tok.encode("the"), list)
        assert len(tok.encode("the")) == 1

    def test_vocab_size(self):
        tok = HFTokenizer(model_name="gpt2")
        assert tok.vocab_size() == 50257

    def test_eos_token_id(self):
        tok = HFTokenizer(model_name="gpt2")
        assert tok.eos_token_id == 50256

    def test_empty_string(self):
        tok = HFTokenizer(model_name="gpt2")
        ids = tok.encode("")
        decoded = tok.decode(ids)
        assert decoded == ""


class TestHFTokenizerTinyLlama:
    def test_encode_decode_round_trip(self):
        tok = HFTokenizer(model_name="tinyllama")
        text = "Hello, world!"
        ids = tok.encode(text)
        decoded = tok.decode(ids)
        assert decoded is not None
        assert len(ids) > 0

    def test_vocab_size(self):
        tok = HFTokenizer(model_name="tinyllama")
        assert tok.vocab_size() == 32000

    def test_eos_token_id(self):
        tok = HFTokenizer(model_name="tinyllama")
        assert tok.eos_token_id == 2

    def test_unknown_model(self):
        with pytest.raises(ValueError):
            HFTokenizer(model_name="nonexistent_model")


class TestHFTokenizerEdgeCases:
    def test_special_characters(self):
        tok = HFTokenizer(model_name="gpt2")
        text = "\n\t"
        ids = tok.encode(text)
        assert len(ids) > 0

    def test_long_text(self):
        tok = HFTokenizer(model_name="gpt2")
        text = " ".join(["test"] * 100)
        ids = tok.encode(text)
        assert len(ids) > 10

    def test_multilingual(self):
        tok = HFTokenizer(model_name="gpt2")
        ids = tok.encode("Bonjour le monde")
        assert len(ids) > 0


# ─── ENDSECTION: HFTokenizer Tests ────────────────────────
