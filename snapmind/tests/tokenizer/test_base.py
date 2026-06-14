import pytest


class TestTokenizerABC:
    def test_cannot_instantiate_directly(self):
        from snapmind.tokenizer.base import TokenizerABC

        with pytest.raises(TypeError):
            TokenizerABC()

    def test_minimal_subclass_works(self):
        from snapmind.tokenizer.base import TokenizerABC

        class MinimalTokenizer(TokenizerABC):
            def encode(self, text):
                return [0]

            def decode(self, ids):
                return "hello"

            def vocab_size(self):
                return 256

        tok = MinimalTokenizer()
        assert tok.encode("test") == [0]
        assert tok.decode([0]) == "hello"
        assert tok.vocab_size() == 256

    def test_registered_via_TOKENIZER(self):
        from snapmind.core.registry import TOKENIZER

        assert "hf" in TOKENIZER
