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


# ANCHOR: SamplingConfig
@dataclass
class SamplingConfig:
    """Configuration for token sampling and generation length."""

    temperature: float = 1.0
    max_tokens: int = 100
    top_k: int | None = None
    top_p: float | None = None
    mirostat_tau: float | None = None
    mirostat_learning_rate: float | None = None


# ENDANCHOR: SamplingConfig


# ANCHOR: EngineConfig
@dataclass
class IndexerConfig:
    """Configuration for the lookahead sparse attention indexer."""

    indexer_type: str = ""
    """Key into the INDEXER registry. Empty string disables the indexer."""
    indexer_layers: tuple[int, ...] = (10, 12, 20)
    """Transformer layers whose hidden states feed the indexer."""
    indexer_interval: int = 64
    """Decode steps between indexer evaluations (τ in the paper)."""
    score_threshold: float = 0.5
    """Sigmoid threshold: chunks scoring >= threshold are hot."""
    score_k: int = 0
    """If > 0, keep only top-k chunks by score (alternative to threshold)."""
    kv_lora_rank: int = 512
    """Compressed KV dimension (matches MLA's kv_lora_rank)."""


@dataclass
class EngineConfig:
    """Configuration for the prefill-decode engine pipeline."""

    max_batch_tokens: int = 4096
    device: str = "auto"
    dtype: str = "bfloat16"
    max_tokens: int = 100


# ENDANCHOR: EngineConfig
# ─── ENDSECTION: Config ─────────────────────────────────
