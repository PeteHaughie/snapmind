import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from snapmind.core.config import EngineConfig, KVCacheConfig, ModelConfig, SamplingConfig
from snapmind.core.registry import Registry, RegistryError
from snapmind.kv_cache.naive import NaiveKVCache
from snapmind.layers.attention.sdpa import create_causal_mask
from snapmind.sampling.greedy import GreedySampler
from snapmind.sampling.mirostat import MirostatSampler
from snapmind.sampling.temperature import TemperatureSampler
from snapmind.sampling.top_k import TopKSampler
from snapmind.sampling.top_p import TopPSampler

# ─── helpers ──────────────────────────────────────────────────────────────

SINGLE_BATCH = 1
MULTI_BATCH = 3
VOCAB_SIZES = st.integers(min_value=2, max_value=32000)
BATCH_SIZES = st.integers(min_value=1, max_value=8)


def logits_strategy(vocab_size: int, batch: int = SINGLE_BATCH) -> st.SearchStrategy[torch.Tensor]:
    return st.builds(
        lambda data: torch.tensor(data, dtype=torch.float32),
        st.lists(
            st.floats(min_value=-50.0, max_value=50.0, allow_nan=False, allow_infinity=False),
            min_size=batch * vocab_size,
            max_size=batch * vocab_size,
        ).map(lambda lst: [lst[i * vocab_size:(i + 1) * vocab_size] for i in range(batch)]),
    )


def valid_token(vocab_size: int, batch: int, output) -> bool:
    if batch == 1:
        return 0 <= output < vocab_size
    return bool(torch.all((output >= 0) & (output < vocab_size)).item())


# ─── Sampler: Greedy ─────────────────────────────────────────────────────

class TestGreedySamplerProperties:
    @given(logits=logits_strategy(100))
    @settings(max_examples=50)
    def test_returns_argmax(self, logits):
        sampler = GreedySampler()
        token = sampler.sample(logits)
        assert token == logits.argmax(dim=-1).item()

    @given(logits=logits_strategy(100, batch=3))
    @settings(max_examples=50)
    def test_batched_returns_argmax(self, logits):
        sampler = GreedySampler()
        token = sampler.sample(logits)
        expected = logits.argmax(dim=-1)
        assert torch.equal(token, expected)

    @given(vocab_size=VOCAB_SIZES, batch=BATCH_SIZES)
    @settings(max_examples=30)
    def test_always_valid_index(self, vocab_size, batch):
        sampler = GreedySampler()
        logits = torch.randn(batch, vocab_size)
        token = sampler.sample(logits)
        assert valid_token(vocab_size, batch, token)

    def test_temperature_ignored(self):
        sampler = GreedySampler()
        logits = torch.tensor([[1.0, 10.0, 5.0]])
        t0 = sampler.sample(logits, temperature=0.0)
        t1 = sampler.sample(logits, temperature=1.0)
        t2 = sampler.sample(logits, temperature=5.0)
        assert t0 == t1 == t2 == 1


# ─── Sampler: Temperature ────────────────────────────────────────────────

class TestTemperatureSamplerProperties:
    @given(logits=logits_strategy(100))
    @settings(max_examples=50)
    def test_temp_zero_returns_argmax(self, logits):
        sampler = TemperatureSampler()
        token = sampler.sample(logits, temperature=0.0)
        assert token == logits.argmax(dim=-1).item()

    @given(logits=logits_strategy(100))
    @settings(max_examples=50)
    def test_temp_near_zero_returns_argmax(self, logits):
        sampler = TemperatureSampler()
        token = sampler.sample(logits, temperature=1e-9)
        assert token == logits.argmax(dim=-1).item()

    @given(vocab_size=VOCAB_SIZES, batch=BATCH_SIZES)
    @settings(max_examples=30)
    def test_always_valid_index(self, vocab_size, batch):
        sampler = TemperatureSampler()
        logits = torch.randn(batch, vocab_size)
        token = sampler.sample(logits, temperature=1.0)
        assert valid_token(vocab_size, batch, token)

    def test_temperature_one_preserves_order_statistically(self):
        sampler = TemperatureSampler()
        logits = torch.full((1, 100), -float("inf"))
        logits[..., 0] = 10.0
        logits[..., 1] = 5.0
        tokens = [sampler.sample(logits, temperature=1.0).item() for _ in range(50)]
        majority = max(set(tokens), key=tokens.count)
        assert majority == 0


# ─── Sampler: Top-K ──────────────────────────────────────────────────────

class TestTopKSamplerProperties:
    @given(logits=logits_strategy(100))
    @settings(max_examples=50)
    def test_temp_zero_returns_argmax(self, logits):
        sampler = TopKSampler()
        token = sampler.sample(logits, temperature=0.0, top_k=50)
        assert token == logits.argmax(dim=-1).item()

    def test_top_k_one_returns_argmax(self):
        sampler = TopKSampler()
        logits = torch.tensor([[1.0, 10.0, 5.0, 2.0, 3.0]])
        token = sampler.sample(logits, temperature=1.0, top_k=1)
        assert token == 1

    @given(vocab_size=VOCAB_SIZES, batch=BATCH_SIZES)
    @settings(max_examples=30)
    def test_always_valid_index(self, vocab_size, batch):
        sampler = TopKSampler()
        logits = torch.randn(batch, vocab_size)
        token = sampler.sample(logits, temperature=1.0, top_k=50)
        assert valid_token(vocab_size, batch, token)

    @given(vocab_size=VOCAB_SIZES, batch=BATCH_SIZES)
    @settings(max_examples=20)
    def test_top_k_clamps_to_vocab_size(self, vocab_size, batch):
        sampler = TopKSampler()
        logits = torch.randn(batch, vocab_size)
        token = sampler.sample(logits, temperature=1.0, top_k=vocab_size + 100)
        assert valid_token(vocab_size, batch, token)

    def test_negative_top_k_still_samples(self):
        sampler = TopKSampler()
        logits = torch.randn(1, 100)
        token = sampler.sample(logits, temperature=1.0, top_k=-1)
        assert valid_token(100, 1, token)


# ─── Sampler: Top-P ──────────────────────────────────────────────────────

class TestTopPSamplerProperties:
    @given(logits=logits_strategy(100))
    @settings(max_examples=50)
    def test_temp_zero_returns_argmax(self, logits):
        sampler = TopPSampler()
        token = sampler.sample(logits, temperature=0.0, top_p=0.9)
        assert token == logits.argmax(dim=-1).item()

    @given(logits=logits_strategy(100))
    @settings(max_examples=50)
    def test_top_p_one_includes_all(self, logits):
        sampler = TopPSampler()
        token = sampler.sample(logits, temperature=1.0, top_p=1.0)
        assert valid_token(100, 1, token)

    @given(vocab_size=VOCAB_SIZES, batch=BATCH_SIZES)
    @settings(max_examples=30)
    def test_always_valid_index(self, vocab_size, batch):
        sampler = TopPSampler()
        logits = torch.randn(batch, vocab_size)
        token = sampler.sample(logits, temperature=1.0, top_p=0.9)
        assert valid_token(vocab_size, batch, token)

    def test_zero_top_p_returns_argmax(self):
        sampler = TopPSampler()
        logits = torch.randn(1, 100)
        token = sampler.sample(logits, temperature=1.0, top_p=0.0)
        assert token == logits.argmax(dim=-1).item()


# ─── Sampler: Mirostat ───────────────────────────────────────────────────

class TestMirostatSamplerProperties:
    @given(logits=logits_strategy(100))
    @settings(max_examples=50)
    def test_temp_zero_returns_argmax(self, logits):
        sampler = MirostatSampler(tau=5.0, learning_rate=0.1)
        token = sampler.sample(logits, temperature=0.0)
        assert token == logits.argmax(dim=-1).item()

    @given(vocab_size=VOCAB_SIZES, batch=BATCH_SIZES)
    @settings(max_examples=30)
    def test_always_valid_index(self, vocab_size, batch):
        sampler = MirostatSampler(tau=5.0, learning_rate=0.1)
        logits = torch.randn(batch, vocab_size)
        token = sampler.sample(logits, temperature=1.0)
        assert valid_token(vocab_size, batch, token)

    @given(vocab_size=VOCAB_SIZES, batch=st.integers(min_value=1, max_value=2))
    @settings(max_examples=20)
    def test_state_updates_after_sample(self, vocab_size, batch):
        sampler = MirostatSampler(tau=5.0, learning_rate=0.1)
        initial = sampler.max_surprise
        logits = torch.randn(batch, vocab_size)
        sampler.sample(logits, temperature=1.0)
        assert sampler.max_surprise != initial or abs(sampler.max_surprise - initial) < 1e-6

    def test_reset_restores_default(self):
        sampler = MirostatSampler(tau=5.0, learning_rate=0.1, max_surprise=2.0)
        logits = torch.randn(1, 100)
        for _ in range(10):
            sampler.sample(logits, temperature=1.0)
        sampler.reset()
        assert sampler.max_surprise == 2.0

    def test_state_clamped(self):
        sampler = MirostatSampler(tau=5.0, learning_rate=100.0, max_surprise=2.0)
        logits = torch.tensor([[1.0, 0.0, 0.0]])
        sampler.sample(logits, temperature=0.1)
        assert 0.1 <= sampler.max_surprise <= 10.0


# ─── Config ──────────────────────────────────────────────────────────────

class TestModelConfigProperties:
    @given(st.integers(min_value=64, max_value=8192))
    @settings(max_examples=20)
    def test_d_ff_inferred(self, d_model):
        config = ModelConfig(d_model=d_model, d_ff=None)
        assert config.d_ff == d_model * 4

    @given(st.integers(min_value=1, max_value=128))
    @settings(max_examples=20)
    def test_n_kv_heads_defaults(self, n_heads):
        config = ModelConfig(n_heads=n_heads, n_kv_heads=None)
        assert config.n_kv_heads == n_heads

    @given(st.integers(min_value=1, max_value=128), st.integers(min_value=1, max_value=32))
    @settings(max_examples=20)
    def test_n_kv_heads_explicit(self, n_heads, n_kv_heads):
        config = ModelConfig(n_heads=n_heads, n_kv_heads=n_kv_heads)
        assert config.n_kv_heads == n_kv_heads

    @given(
        st.integers(min_value=64, max_value=4096),
        st.integers(min_value=1, max_value=32),
        st.integers(min_value=1, max_value=32),
    )
    @settings(max_examples=20)
    def test_round_trip(self, d_model, n_heads, n_layers):
        original = ModelConfig(d_model=d_model, n_heads=n_heads, n_layers=n_layers)
        restored = ModelConfig.from_dict(original.to_dict())
        assert restored == original

    def test_minimal_defaults(self):
        config = ModelConfig()
        assert config.d_ff == config.d_model * 4
        assert config.n_kv_heads == config.n_heads
        assert config.attention_type == "sdpa"
        assert config.pe_type == "rope"


class TestSamplingConfigProperties:
    @given(st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False))
    @settings(max_examples=20)
    def test_temperature_accepts_any_non_negative(self, temperature):
        config = SamplingConfig(temperature=temperature)
        assert config.temperature == temperature

    @given(st.integers(min_value=1, max_value=4096))
    @settings(max_examples=20)
    def test_max_tokens_positive(self, max_tokens):
        config = SamplingConfig(max_tokens=max_tokens)
        assert config.max_tokens == max_tokens

    @given(st.integers(min_value=1, max_value=100).map(lambda x: x if x % 2 == 0 else None))
    @settings(max_examples=10)
    def test_top_k_none_or_positive(self, top_k):
        config = SamplingConfig(top_k=top_k)
        assert config.top_k == top_k


class TestKVCacheConfigProperties:
    @given(st.integers(min_value=128, max_value=65536))
    @settings(max_examples=10)
    def test_max_seq_len_positive(self, max_seq_len):
        config = KVCacheConfig(max_seq_len=max_seq_len)
        assert config.max_seq_len == max_seq_len


class TestEngineConfigProperties:
    @given(st.integers(min_value=128, max_value=65536))
    @settings(max_examples=10)
    def test_max_batch_tokens_positive(self, max_batch_tokens):
        config = EngineConfig(max_batch_tokens=max_batch_tokens)
        assert config.max_batch_tokens == max_batch_tokens


# ─── Registry ────────────────────────────────────────────────────────────

class TestRegistryProperties:
    @given(st.text(min_size=1, max_size=20, alphabet="abcdefghijklmnopqrstuvwxyz_"))
    @settings(max_examples=20)
    def test_register_and_create_round_trip(self, name):
        reg = Registry("test", expected_type=object)

        cls = type(name, (object,), {"__init__": lambda self: None})
        reg.register(name, cls)
        instance = reg.create(name)
        assert isinstance(instance, cls)

    @given(st.text(min_size=1, max_size=20, alphabet="abcdefghijklmnopqrstuvwxyz_"))
    @settings(max_examples=20)
    def test_list_includes_registered(self, name):
        reg = Registry("test", expected_type=object)
        cls = type(name, (object,), {"__init__": lambda self: None})
        reg.register(name, cls)
        assert name in reg.list()

    @given(st.text(min_size=1, max_size=20, alphabet="abcdefghijklmnopqrstuvwxyz_"))
    @settings(max_examples=20)
    def test_contains_after_register(self, name):
        reg = Registry("test", expected_type=object)
        cls = type(name, (object,), {"__init__": lambda self: None})
        reg.register(name, cls)
        assert name in reg

    def test_register_nonexistent_create_raises(self):
        reg = Registry("test", expected_type=object)
        with pytest.raises(RegistryError):
            reg.create("nope")

    @given(st.lists(st.text(min_size=1, max_size=10, alphabet="abc"), unique=True, min_size=1, max_size=5))
    @settings(max_examples=10)
    def test_list_returns_copy(self, names):
        reg = Registry("test", expected_type=object)
        for n in names:
            reg.register(n, type(n, (object,), {"__init__": lambda self: None}))
        lst = reg.list()
        lst.append("extra")
        assert "extra" not in reg.list()
        assert set(names) == set(reg.list())


# ─── Causal Mask ─────────────────────────────────────────────────────────

class TestCausalMaskProperties:
    @given(st.integers(min_value=1, max_value=1024))
    @settings(max_examples=20)
    def test_shape(self, seq_len):
        mask = create_causal_mask(seq_len)
        assert mask.shape == (seq_len, seq_len)

    @given(st.integers(min_value=1, max_value=512))
    @settings(max_examples=20, deadline=None)
    def test_lower_triangular_zero(self, seq_len):
        mask = create_causal_mask(seq_len)
        lower_bool = torch.tril(torch.ones_like(mask, dtype=torch.bool), diagonal=0)
        assert torch.all(mask[lower_bool] == 0.0)

    @given(st.integers(min_value=1, max_value=512))
    @settings(max_examples=20, deadline=None)
    def test_upper_triangular_neg_inf(self, seq_len):
        mask = create_causal_mask(seq_len)
        upper_bool = torch.triu(torch.ones_like(mask, dtype=torch.bool), diagonal=1)
        if upper_bool.any():
            assert torch.all(mask[upper_bool] == float("-inf"))

    @given(st.integers(min_value=1, max_value=1024))
    @settings(max_examples=20)
    def test_diagonal_zero(self, seq_len):
        mask = create_causal_mask(seq_len)
        assert torch.all(torch.diag(mask) == 0.0)


# ─── KV Cache ────────────────────────────────────────────────────────────

class TestNaiveKVCacheProperties:
    N_LAYERS = st.integers(min_value=1, max_value=4)
    N_HEADS = st.integers(min_value=1, max_value=8)
    HEAD_DIM = st.integers(min_value=16, max_value=128)
    SEQ_POS = st.integers(min_value=0, max_value=256)
    MAX_SEQ = 1024

    @given(
        layer=N_LAYERS,
        n_heads=N_HEADS,
        head_dim=HEAD_DIM,
    )
    @settings(max_examples=20)
    def test_store_then_fetch(self, layer, n_heads, head_dim):
        cache = NaiveKVCache(max_seq_len=self.MAX_SEQ, n_layers=layer + 1, n_heads=n_heads, head_dim=head_dim)
        k = torch.randn(1, n_heads, 1, head_dim)
        v = torch.randn(1, n_heads, 1, head_dim)
        cache.store(layer, k, v, 0)
        fetched_k, fetched_v = cache.fetch(layer)
        assert torch.equal(fetched_k, k)
        assert torch.equal(fetched_v, v)

    @given(
        layer=N_LAYERS,
        n_heads=N_HEADS,
        head_dim=HEAD_DIM,
        extra_tokens=st.integers(min_value=1, max_value=5),
    )
    @settings(max_examples=20)
    def test_consecutive_stores_concatenate(self, layer, n_heads, head_dim, extra_tokens):
        cache = NaiveKVCache(max_seq_len=self.MAX_SEQ, n_layers=layer + 1, n_heads=n_heads, head_dim=head_dim)
        for pos in range(extra_tokens):
            k = torch.randn(1, n_heads, 1, head_dim)
            v = torch.randn(1, n_heads, 1, head_dim)
            cache.store(layer, k, v, pos)
        fetched_k, fetched_v = cache.fetch(layer)
        assert fetched_k.shape[-2] == extra_tokens
        assert fetched_v.shape[-2] == extra_tokens

    @given(
        layer=N_LAYERS,
        n_heads=N_HEADS,
        head_dim=HEAD_DIM,
        extra_tokens=st.integers(min_value=2, max_value=6),
    )
    @settings(max_examples=20)
    def test_evict_keeps_correct_count(self, layer, n_heads, head_dim, extra_tokens):
        cache = NaiveKVCache(max_seq_len=self.MAX_SEQ, n_layers=layer + 1, n_heads=n_heads, head_dim=head_dim)
        for pos in range(extra_tokens):
            k = torch.randn(1, n_heads, 1, head_dim)
            v = torch.randn(1, n_heads, 1, head_dim)
            cache.store(layer, k, v, pos)
        keep = max(1, extra_tokens - 1)
        cache.evict(keep)
        fetched_k, _ = cache.fetch(layer)
        assert fetched_k.shape[-2] == keep

    @given(
        n_heads=N_HEADS,
        head_dim=HEAD_DIM,
    )
    @settings(max_examples=10)
    def test_reset_clears_all_layers(self, n_heads, head_dim):
        n_layers = 3
        cache = NaiveKVCache(max_seq_len=self.MAX_SEQ, n_layers=n_layers, n_heads=n_heads, head_dim=head_dim)
        for layer in range(n_layers):
            k = torch.randn(1, n_heads, 1, head_dim)
            v = torch.randn(1, n_heads, 1, head_dim)
            cache.store(layer, k, v, 0)
        cache.reset()
        for layer in range(n_layers):
            fetched_k, _ = cache.fetch(layer)
            assert fetched_k.numel() == 0

    @given(
        n_heads=N_HEADS,
        head_dim=HEAD_DIM,
    )
    @settings(max_examples=10)
    def test_fetch_empty_returns_empty(self, n_heads, head_dim):
        cache = NaiveKVCache(max_seq_len=self.MAX_SEQ, n_layers=3, n_heads=n_heads, head_dim=head_dim)
        k, v = cache.fetch(0)
        assert k.numel() == 0
        assert v.numel() == 0

    @given(
        extra_tokens=st.integers(min_value=1, max_value=5),
        n_heads=N_HEADS,
        head_dim=HEAD_DIM,
    )
    @settings(max_examples=10)
    def test_memory_usage_reflects_stored_tokens(self, extra_tokens, n_heads, head_dim):
        n_layers = 2
        cache = NaiveKVCache(max_seq_len=self.MAX_SEQ, n_layers=n_layers, n_heads=n_heads, head_dim=head_dim)
        for pos in range(extra_tokens):
            k = torch.randn(1, n_heads, 1, head_dim)
            v = torch.randn(1, n_heads, 1, head_dim)
            for layer in range(n_layers):
                cache.store(layer, k, v, pos)
        usage = cache.memory_usage()
        expected_bytes = n_layers * extra_tokens * 2 * n_heads * head_dim * 4
        assert usage["num_tokens"] == n_layers * extra_tokens
        assert usage["total_bytes"] == expected_bytes


# ─── GQA attention shape invariants ──────────────────────────────────────

class TestGQAShapeProperties:
    @given(
        batch=st.integers(min_value=1, max_value=4),
        seq_len=st.integers(min_value=1, max_value=64),
        d_model=st.sampled_from([64, 128, 256]),
        n_heads=st.sampled_from([4, 8]),
    )
    @settings(max_examples=20)
    def test_forward_output_shape(self, batch, seq_len, d_model, n_heads):
        from snapmind.layers.attention.gqa import GroupedQueryAttention

        attn = GroupedQueryAttention(d_model=d_model, n_heads=n_heads, n_kv_heads=n_heads // 2)
        x = torch.randn(batch, seq_len, d_model)
        out, weights = attn(x)
        assert out.shape == (batch, seq_len, d_model)
        assert weights.shape == (batch, n_heads, seq_len, seq_len)

    @given(
        batch=st.integers(min_value=1, max_value=4),
        seq_len=st.integers(min_value=1, max_value=64),
        d_model=st.sampled_from([64, 128, 256]),
        n_heads=st.sampled_from([4, 8]),
    )
    @settings(max_examples=20)
    def test_gqa_with_mask_shape(self, batch, seq_len, d_model, n_heads):
        from snapmind.layers.attention.gqa import GroupedQueryAttention

        attn = GroupedQueryAttention(d_model=d_model, n_heads=n_heads, n_kv_heads=n_heads // 2)
        x = torch.randn(batch, seq_len, d_model)
        mask = create_causal_mask(seq_len)
        out, weights = attn(x, mask=mask)
        assert out.shape == (batch, seq_len, d_model)
        assert weights.shape == (batch, n_heads, seq_len, seq_len)
