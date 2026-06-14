# ─── SECTION: HuggingFace Tokenizer ─────────────────────
import tiktoken
from snapmind.core.registry import TOKENIZER
from snapmind.tokenizer.base import TokenizerABC


_EOS_TOKEN_IDS = {"gpt2": 50256}


# ANCHOR: HFTokenizer
@TOKENIZER.register("hf")
class HFTokenizer(TokenizerABC):
    def __init__(self, model_name: str = "gpt2"):
        self.model_name = model_name
        self._tokenizer = tiktoken.get_encoding("gpt2")

    def encode(self, text: str):
        return self._tokenizer.encode(text)

    def decode(self, ids):
        return self._tokenizer.decode(ids)

    def vocab_size(self) -> int:
        return self._tokenizer.n_vocab

    @property
    def eos_token_id(self) -> int:
        return _EOS_TOKEN_IDS.get(self.model_name, 50256)
# ENDANCHOR: HFTokenizer
# ─── ENDSECTION: HuggingFace Tokenizer ──────────────────
