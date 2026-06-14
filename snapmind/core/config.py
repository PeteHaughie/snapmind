# ─── SECTION: Config ────────────────────────────────────
from dataclasses import dataclass, field, asdict
from typing import Optional


# ANCHOR: ModelConfig
@dataclass
class ModelConfig:
    model_type: str = "llama"
    d_model: int = 4096
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    n_layers: int = 32
    d_ff: Optional[int] = None
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

    def __post_init__(self):
        if self.d_ff is None:
            self.d_ff = self.d_model * 4
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)


# ANCHOR: KVCacheConfig
@dataclass
class KVCacheConfig:
    max_seq_len: int = 8192
    dtype: str = "bfloat16"
    device: str = "auto"
# ENDANCHOR: KVCacheConfig


# ANCHOR: EngineConfig
@dataclass
class EngineConfig:
    max_batch_tokens: int = 4096
    device: str = "auto"
    dtype: str = "bfloat16"
# ENDANCHOR: EngineConfig
# ─── ENDSECTION: Config ─────────────────────────────────
