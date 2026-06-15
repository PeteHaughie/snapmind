# ─── SECTION: HuggingFace Tokenizer ─────────────────────
from typing import Any

from snapmind.core.architecture import ARCHITECTURE
from snapmind.core.registry import TOKENIZER
from snapmind.tokenizer.base import TokenizerABC


# ANCHOR: HFTokenizer
@TOKENIZER.register("hf")
class HFTokenizer(TokenizerABC):
    _tokenizer: Any

    def __init__(self, model_name: str = "gpt2"):
        self.model_name = model_name
        arch = ARCHITECTURE.get(model_name) if model_name in ARCHITECTURE else None
        hf_id = arch.tokenizer_hf_repo if arch and arch.tokenizer_hf_repo else (arch.hf_repo if arch else None)

        if model_name == "gpt2":
            import tiktoken

            self._tokenizer = tiktoken.get_encoding("gpt2")
            self._eos_token_id = self._tokenizer.eot_token
        else:
            from tokenizers import Tokenizer

            if hf_id is None:
                raise ValueError(f"No tokenizer mapping for '{model_name}'")
            self._tokenizer = Tokenizer.from_pretrained(hf_id)
            self._eos_token_id = (
                self._tokenizer.token_to_id("</s>")
                or self._tokenizer.token_to_id("<|end_of_text|>")
                or 2
            )

    def encode(self, text: str):
        if self.model_name == "gpt2":
            return self._tokenizer.encode(text)
        return self._tokenizer.encode(text).ids

    def decode(self, ids) -> str:
        return self._tokenizer.decode(ids)

    def vocab_size(self) -> int:
        if self.model_name == "gpt2":
            return self._tokenizer.n_vocab
        return self._tokenizer.get_vocab_size()

    @property
    def eos_token_id(self) -> int:
        return self._eos_token_id


# ENDANCHOR: HFTokenizer
# ─── ENDSECTION: HuggingFace Tokenizer ──────────────────
