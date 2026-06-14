# ─── SECTION: Tokenizer ABC ─────────────────────────────
import abc


# ANCHOR: TokenizerABC
class TokenizerABC(abc.ABC):
    @abc.abstractmethod
    def encode(self, text: str) -> list[int]: ...

    @abc.abstractmethod
    def decode(self, ids: list[int]) -> str: ...

    @abc.abstractmethod
    def vocab_size(self) -> int: ...


# ENDANCHOR: TokenizerABC
# ─── ENDSECTION: Tokenizer ABC ──────────────────────────
