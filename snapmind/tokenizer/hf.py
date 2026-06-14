# ─── SECTION: HuggingFace Tokenizer ─────────────────────
from snapmind.core.registry import TOKENIZER
from snapmind.tokenizer.base import TokenizerABC

_HF_TOKENIZER_ID = {
    "gpt2": "openai-community/gpt2",
    "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "mistral": "mistralai/Mistral-7B-v0.1",
    "ministral-3-3b": "mistralai/Ministral-3-3B-Base-2512",
}


# ANCHOR: HFTokenizer
@TOKENIZER.register("hf")
class HFTokenizer(TokenizerABC):
    def __init__(self, model_name: str = "gpt2"):
        self.model_name = model_name
        if model_name == "gpt2":
            import tiktoken

            self._tokenizer = tiktoken.get_encoding("gpt2")
        else:
            from tokenizers import Tokenizer

            hf_id = _HF_TOKENIZER_ID.get(model_name)
            if hf_id is None:
                raise ValueError(f"No tokenizer mapping for '{model_name}'")
            self._tokenizer = Tokenizer.from_pretrained(hf_id)  # type: ignore[assignment]

    def encode(self, text: str):
        if self.model_name == "gpt2":
            return self._tokenizer.encode(text)
        return self._tokenizer.encode(text).ids  # type: ignore[attr-defined]

    def decode(self, ids) -> str:
        return self._tokenizer.decode(ids)

    def vocab_size(self) -> int:
        if self.model_name == "gpt2":
            return self._tokenizer.n_vocab
        return self._tokenizer.get_vocab_size()  # type: ignore[attr-defined]

    @property
    def eos_token_id(self) -> int:
        if self.model_name == "gpt2":
            return 50256
        return 2


# ENDANCHOR: HFTokenizer
# ─── ENDSECTION: HuggingFace Tokenizer ──────────────────
