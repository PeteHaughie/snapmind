# ─── SECTION: InferenceLogger Tests ───────────────────────
import time

from snapmind.core.logging import InferenceLogger


class TestInferenceLoggerBasic:
    def test_initial_state(self):
        logger = InferenceLogger()
        assert logger.num_prefill_tokens == 0
        assert logger.num_decode_tokens == 0
        assert logger.total_time == 0.0
        assert logger.ttft is None
        assert logger.tokens_per_second is None

    def test_start_initializes(self):
        logger = InferenceLogger()
        logger.start()
        assert logger._start_time is not None

    def test_log_prefill(self):
        logger = InferenceLogger()
        logger.start()
        logger.log_prefill(128)
        assert logger.num_prefill_tokens == 128

    def test_log_token(self):
        logger = InferenceLogger()
        logger.start()
        logger.log_token()
        assert logger.num_decode_tokens == 1

    def test_summary_keys(self):
        logger = InferenceLogger()
        logger.start()
        logger.log_prefill(128)
        logger.log_token()
        logger.log_token()
        s = logger.summary()
        assert "num_prefill_tokens" in s
        assert "num_decode_tokens" in s
        assert "total_tokens" in s
        assert "total_time_seconds" in s
        assert "ttft_seconds" in s
        assert "tokens_per_second" in s


class TestInferenceLoggerTiming:
    def test_ttft_after_prefill(self):
        logger = InferenceLogger()
        logger.start()
        time.sleep(0.01)
        logger.log_prefill(128)
        assert logger.ttft is not None
        assert logger.ttft > 0.005

    def test_tokens_per_second(self):
        logger = InferenceLogger()
        logger.start()
        logger.log_prefill(0)
        logger.log_token()
        time.sleep(0.05)
        logger.log_token()
        # After 2+ tokens, compute speed of decode phase
        tps = logger.tokens_per_second
        assert tps is not None
        assert tps > 0
        assert tps < 100  # sanity: not unrealistically fast


class TestInferenceLoggerReset:
    def test_reset_clears(self):
        logger = InferenceLogger()
        logger.start()
        logger.log_prefill(128)
        logger.log_token()
        logger.reset()
        assert logger.num_prefill_tokens == 0
        assert logger.num_decode_tokens == 0
        assert logger.ttft is None


# ─── ENDSECTION: InferenceLogger Tests ────────────────────
