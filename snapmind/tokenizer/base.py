# ─── SECTION: Tokenizer ABC ─────────────────────────────
import abc


# ANCHOR: TokenizerABC
class TokenizerABC(abc.ABC):
    """Base class for tokenizers (HF, tiktoken, sentencepiece, …).

    Subclasses implement :meth:`encode`, :meth:`decode`, and :meth:`vocab_size`.
    """

    @abc.abstractmethod
    def encode(self, text: str) -> list[int]:
        """Tokenize *text* into a list of token IDs."""

    @abc.abstractmethod
    def decode(self, ids: list[int]) -> str:
        """Convert a list of token IDs back into a string."""

    @abc.abstractmethod
    def vocab_size(self) -> int:
        """Return the size of the vocabulary (number of unique tokens)."""


# ENDANCHOR: TokenizerABC
# ─── ENDSECTION: Tokenizer ABC ──────────────────────────
