# ─── SECTION: Inference Logger ────────────────────────────
import time


# ANCHOR: InferenceLogger
class InferenceLogger:
    def __init__(self):
        self._start_time: float | None = None
        self._prefill_end_time: float | None = None
        self._token_times: list[float] = []
        self._num_prefill_tokens: int = 0
        self._num_decode_tokens: int = 0

    def start(self) -> None:
        self._start_time = time.monotonic()

    def log_prefill(self, num_tokens: int) -> None:
        self._prefill_end_time = time.monotonic()
        self._num_prefill_tokens = num_tokens

    def log_token(self) -> None:
        self._token_times.append(time.monotonic())
        self._num_decode_tokens += 1

    @property
    def num_prefill_tokens(self) -> int:
        return self._num_prefill_tokens

    @property
    def num_decode_tokens(self) -> int:
        return self._num_decode_tokens

    @property
    def total_time(self) -> float:
        if self._start_time is None:
            return 0.0
        return (self._token_times[-1] if self._token_times else time.monotonic()) - self._start_time

    @property
    def ttft(self) -> float | None:
        if self._start_time is None or self._prefill_end_time is None:
            return None
        return self._prefill_end_time - self._start_time

    @property
    def tokens_per_second(self) -> float | None:
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
        return {
            "num_prefill_tokens": self._num_prefill_tokens,
            "num_decode_tokens": self._num_decode_tokens,
            "total_tokens": self._num_prefill_tokens + self._num_decode_tokens,
            "total_time_seconds": round(self.total_time, 3),
            "ttft_seconds": round(self.ttft, 3) if self.ttft is not None else None,
            "tokens_per_second": round(self.tokens_per_second, 1) if self.tokens_per_second is not None else None,
        }

    def print_summary(self) -> None:
        s = self.summary()
        print(f"  Prefill: {s['num_prefill_tokens']} tokens")
        print(f"  Decode:  {s['num_decode_tokens']} tokens")
        print(f"  Total:   {s['total_tokens']} tokens in {s['total_time_seconds']}s")
        if s["ttft_seconds"] is not None:
            print(f"  TTFT:    {s['ttft_seconds']}s")
        if s["tokens_per_second"] is not None:
            print(f"  Speed:   {s['tokens_per_second']} tok/s")

    def reset(self) -> None:
        self._start_time = None
        self._prefill_end_time = None
        self._token_times = []
        self._num_prefill_tokens = 0
        self._num_decode_tokens = 0


# ENDANCHOR: InferenceLogger
# ─── ENDSECTION: Inference Logger ─────────────────────────
