# ─── SECTION: Config ────────────────────────────────────
from dataclasses import asdict, dataclass


# ANCHOR: ModelConfig
@dataclass
class ModelConfig:
    """Hyperparameters for a transformer model architecture.

    Fields map to standard transformer config keys (d_model, n_heads, n_layers, …).
    ``d_ff`` and ``n_kv_heads`` default from ``d_model`` and ``n_heads`` respectively.
    """

    model_type: str = "llama"
    d_model: int = 4096
    n_heads: int = 32
    n_kv_heads: int | None = None
    n_layers: int = 32
    d_ff: int | None = None
    vocab_size: int = 32000
    max_seq_len: int = 8192
    norm_eps: float = 1e-5
    attention_type: str = "sdpa"
    pe_type: str = "rope"
    norm_type: str = "rmsnorm"
    activation_type: str = "silu"
    kv_cache_type: str = "naive"
    dropout: float = 0.0
    rope_theta: float = 10000.0
    head_dim: int | None = None
    tie_word_embeddings: bool = False
    window_size: int = 4096

    def __post_init__(self):
        if self.d_ff is None:
            self.d_ff = self.d_model * 4
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads

    def to_dict(self) -> dict:
        """Serialize to a plain dict (via ``dataclasses.asdict``)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict):
        """Deserialize from a plain dict."""
        return cls(**data)


# ANCHOR: KVCacheConfig
@dataclass
class KVCacheConfig:
    """Configuration for KV cache allocation and device placement."""

    max_seq_len: int = 8192
    dtype: str = "bfloat16"
    device: str = "auto"


# ENDANCHOR: KVCacheConfig


# ANCHOR: EngineConfig
@dataclass
class EngineConfig:
    """Configuration for the prefill-decode engine pipeline."""

    max_batch_tokens: int = 4096
    device: str = "auto"
    dtype: str = "bfloat16"


# ENDANCHOR: EngineConfig
# ─── ENDSECTION: Config ─────────────────────────────────
