# ─── SECTION: Inference Logger ────────────────────────────
import time


# ANCHOR: InferenceLogger
class InferenceLogger:
    """Tracks timing and token counts for a single inference run.

    Usage::

        logger = InferenceLogger()
        logger.start()
        # … run prefill …
        logger.log_prefill(num_tokens)
        # … run decode loop …
        for _ in range(n_tokens):
            logger.log_token()
        logger.print_summary()
    """

    def __init__(self):
        self._start_time: float | None = None
        self._prefill_end_time: float | None = None
        self._token_times: list[float] = []
        self._num_prefill_tokens: int = 0
        self._num_decode_tokens: int = 0

    def start(self) -> None:
        """Begin the timer (call before prefill)."""
        self._start_time = time.monotonic()

    def log_prefill(self, num_tokens: int) -> None:
        """Record the end of the prefill phase.

        Args:
            num_tokens: Number of prompt tokens processed.
        """
        self._prefill_end_time = time.monotonic()
        self._num_prefill_tokens = num_tokens

    def log_token(self) -> None:
        """Record a single decode token (calls ``time.monotonic`` internally)."""
        self._token_times.append(time.monotonic())
        self._num_decode_tokens += 1

    @property
    def num_prefill_tokens(self) -> int:
        """Number of prompt tokens from the last ``log_prefill`` call."""
        return self._num_prefill_tokens

    @property
    def num_decode_tokens(self) -> int:
        """Number of decode tokens logged so far."""
        return self._num_decode_tokens

    @property
    def total_time(self) -> float:
        """Wall-clock seconds from ``start()`` to the last logged token (or now if none)."""
        if self._start_time is None:
            return 0.0
        return (self._token_times[-1] if self._token_times else time.monotonic()) - self._start_time

    @property
    def ttft(self) -> float | None:
        """Time-to-first-token (prefill duration), or ``None`` if prefill not logged."""
        if self._start_time is None or self._prefill_end_time is None:
            return None
        return self._prefill_end_time - self._start_time

    @property
    def tokens_per_second(self) -> float | None:
        """Decode throughput (tok/s), excluding TTFT, or ``None`` if fewer than 2 decode tokens."""
        if self._num_decode_tokens == 0 or self._token_times is None or len(self._token_times) < 2:
            if self._num_decode_tokens > 0 and self._start_time is not None:
                elapsed = (self._token_times[-1] if self._token_times else time.monotonic()) - self._start_time
                if elapsed > 0:
                    return self._num_decode_tokens / elapsed
            return None
        decode_start = self._token_times[0]
        decode_end = self._token_times[-1]
        elapsed = decode_end - decode_start
        if elapsed <= 0:
            return None
        return (self._num_decode_tokens - 1) / elapsed

    def summary(self) -> dict:
        """Return a dict of all metrics (prefill/decode counts, TTFT, tok/s)."""
        return {
            "num_prefill_tokens": self._num_prefill_tokens,
            "num_decode_tokens": self._num_decode_tokens,
            "total_tokens": self._num_prefill_tokens + self._num_decode_tokens,
            "total_time_seconds": round(self.total_time, 3),
            "ttft_seconds": round(self.ttft, 3) if self.ttft is not None else None,
            "tokens_per_second": round(self.tokens_per_second, 1) if self.tokens_per_second is not None else None,
        }

    def print_summary(self) -> None:
        """Print a human-readable summary to stdout."""
        s = self.summary()
        print(f"  Prefill: {s['num_prefill_tokens']} tokens")
        print(f"  Decode:  {s['num_decode_tokens']} tokens")
        print(f"  Total:   {s['total_tokens']} tokens in {s['total_time_seconds']}s")
        if s["ttft_seconds"] is not None:
            print(f"  TTFT:    {s['ttft_seconds']}s")
        if s["tokens_per_second"] is not None:
            print(f"  Speed:   {s['tokens_per_second']} tok/s")

    def reset(self) -> None:
        """Clear all state so the logger can be reused."""
        self._start_time = None
        self._prefill_end_time = None
        self._token_times = []
        self._num_prefill_tokens = 0
        self._num_decode_tokens = 0


# ENDANCHOR: InferenceLogger
# ─── ENDSECTION: Inference Logger ─────────────────────────
