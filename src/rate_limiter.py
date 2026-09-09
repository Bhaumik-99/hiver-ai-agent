"""
Rate limiter and retry manager for LLM API calls.
Enforces requests-per-minute, tokens-per-minute (TPM), and minimum interval
constraints to prevent HTTP 429 errors on rate-limited providers (e.g. Groq 8k TPM).
"""

import os
import time
import threading
from collections import deque


class GroqRateLimitExceeded(RuntimeError):
    """Raised when Groq rate limits cannot be resolved within max retries."""
    pass


class RateLimiter:
    """
    Sliding-window rate limiter with token tracking and bounded exponential backoff.
    """

    def __init__(self,
                 max_tpm: int = None,
                 min_interval: float = None,
                 max_retries: int = None):
        self.max_tpm = max_tpm or int(os.environ.get("GROQ_MAX_TPM", 7000))
        self.min_interval = (
            min_interval if min_interval is not None
            else float(os.environ.get("GROQ_MIN_REQUEST_INTERVAL", 2.0))
        )
        self.max_retries = max_retries or int(os.environ.get("GROQ_MAX_RETRIES", 3))

        self._lock = threading.Lock()
        self._last_request_time = 0.0
        self._token_history = deque()

    def configure(self, max_tpm: int = None, min_interval: float = None, max_retries: int = None):
        """Update rate limiter settings dynamically."""
        with self._lock:
            if max_tpm is not None:
                self.max_tpm = max_tpm
            if min_interval is not None:
                self.min_interval = min_interval
            if max_retries is not None:
                self.max_retries = max_retries

    def acquire(self, estimated_tokens: int = 150):
        """
        Block until it is safe to make an API request under both
        min_interval and max_tpm limits.
        """
        with self._lock:
            now = time.time()

            elapsed = now - self._last_request_time
            if elapsed < self.min_interval:
                sleep_time = self.min_interval - elapsed
                time.sleep(sleep_time)
                now = time.time()

            cutoff = now - 60.0
            while self._token_history and self._token_history[0][0] < cutoff:
                self._token_history.popleft()

            current_tpm = sum(tok for _, tok in self._token_history)
            if current_tpm + estimated_tokens > self.max_tpm:
                needed_free = (current_tpm + estimated_tokens) - self.max_tpm
                freed = 0
                wait_until = now
                for ts, tok in self._token_history:
                    freed += tok
                    wait_until = ts + 60.0 + 0.1
                    if freed >= needed_free:
                        break
                wait_time = max(0.1, wait_until - now)
                time.sleep(wait_time)
                now = time.time()
                cutoff = now - 60.0
                while self._token_history and self._token_history[0][0] < cutoff:
                    self._token_history.popleft()

            self._token_history.append((now, estimated_tokens))
            self._last_request_time = now

    def handle_429(self, attempt: int, retry_after: str = None) -> float:
        """
        Compute wait time for an HTTP 429 response.
        Uses server-reported Retry-After if present, else bounded exponential backoff.
        Raises GroqRateLimitExceeded if attempt >= max_retries.
        """
        if attempt >= self.max_retries:
            raise GroqRateLimitExceeded(
                f"Groq unavailable after {self.max_retries} attempts; example marked failed."
            )

        wait = None
        if retry_after:
            try:
                wait = float(retry_after)
            except (ValueError, TypeError):
                wait = None

        if wait is None:
            wait = 2.0 ** (attempt + 1)

        wait = min(30.0, max(1.0, wait)) + 0.5
        return wait

    def observe_response(self, headers: dict) -> None:
        """Parse server rate-limit headers to calibrate remaining budget."""
        if not headers:
            return
        rem = headers.get("x-ratelimit-remaining-tokens")
        if rem is not None:
            try:
                rem_val = float(rem)
                if rem_val < 500:
                    rst = headers.get("x-ratelimit-reset-tokens")
                    if rst:
                        wait = self._parse_duration(rst)
                        if wait > 0:
                            time.sleep(min(15.0, wait + 0.2))
            except (ValueError, TypeError):
                pass

    @staticmethod
    def _parse_duration(text: str) -> float:
        """Parse duration strings like '1m2.5s', '500ms', '3.5s' into seconds."""
        if not text:
            return 0.0
        total, num = 0.0, ""
        i = 0
        while i < len(text):
            c = text[i]
            if c.isdigit() or c == ".":
                num += c
                i += 1
            elif text[i:i + 2] == "ms":
                total += float(num or 0) / 1000.0
                num = ""
                i += 2
            elif c == "s":
                total += float(num or 0)
                num = ""
                i += 1
            elif c == "m":
                total += float(num or 0) * 60.0
                num = ""
                i += 1
            else:
                i += 1
        return total


default_rate_limiter = RateLimiter()
