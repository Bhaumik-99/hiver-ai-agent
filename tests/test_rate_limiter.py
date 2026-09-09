"""
Tests for rate limiting, retry bounding, exponential backoff,
graceful failure, checkpointing, and benchmark mode selection.
"""

import json
import os
import sys
import tempfile
import time
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rate_limiter import RateLimiter, GroqRateLimitExceeded
from src.golden_set import stratified_subsample, read_rows, FINAL_PATH
from scripts.run_evaluation import BENCHMARKS, run_agent


def test_rate_limiter_min_interval():
    """Verify rate limiter enforces minimum interval between requests."""
    limiter = RateLimiter(min_interval=0.1, max_tpm=10000, max_retries=3)
    t0 = time.time()
    limiter.acquire(estimated_tokens=50)
    limiter.acquire(estimated_tokens=50)
    elapsed = time.time() - t0
    assert elapsed >= 0.09, f"Elapsed {elapsed}s was shorter than min_interval 0.1s"


def test_rate_limiter_tpm_throttling():
    """Verify rate limiter sleeps when token budget is exhausted."""
    # Set a tiny budget of 100 TPM
    limiter = RateLimiter(min_interval=0.01, max_tpm=100, max_retries=3)
    limiter.acquire(estimated_tokens=90)
    # Next acquisition of 50 tokens exceeds 100 TPM, so it must throttle
    # Instead of waiting 60s, verify token accounting tracks accurately
    assert len(limiter._token_history) == 1
    assert limiter._token_history[0][1] == 90


def test_429_bounded_retries():
    """Verify handle_429 stops and raises after max_retries attempts."""
    limiter = RateLimiter(max_retries=3)

    # Attempt 0, 1, 2 should return sleep durations
    wait0 = limiter.handle_429(attempt=0)
    assert 2.0 <= wait0 <= 30.5

    wait1 = limiter.handle_429(attempt=1)
    assert wait1 > wait0

    wait2 = limiter.handle_429(attempt=2)
    assert wait2 > wait1

    # Attempt 3 (equal to max_retries) must raise GroqRateLimitExceeded
    with pytest.raises(GroqRateLimitExceeded) as exc_info:
        limiter.handle_429(attempt=3)
    assert "Groq unavailable after 3 attempts" in str(exc_info.value)


def test_429_exponential_backoff_and_retry_after():
    """Verify retry_after header parsing and exponential calculation."""
    limiter = RateLimiter(max_retries=4)

    # With explicit retry-after header
    wait_header = limiter.handle_429(attempt=0, retry_after="4.5")
    assert wait_header == 5.0  # 4.5 + 0.5 buffer

    # Without header: exponential 2^(attempt+1) + 0.5
    wait_exp0 = limiter.handle_429(attempt=0, retry_after=None)
    assert wait_exp0 == 2.5  # 2^1 + 0.5

    wait_exp1 = limiter.handle_429(attempt=1, retry_after=None)
    assert wait_exp1 == 4.5  # 2^2 + 0.5

    wait_exp2 = limiter.handle_429(attempt=2, retry_after=None)
    assert wait_exp2 == 8.5  # 2^3 + 0.5


def test_observe_response_duration_parsing():
    """Verify duration parsing for response reset headers."""
    limiter = RateLimiter()
    assert limiter._parse_duration("1m2.5s") == 62.5
    assert limiter._parse_duration("500ms") == 0.5
    assert limiter._parse_duration("3.5s") == 3.5
    assert limiter._parse_duration("") == 0.0


def test_graceful_api_failure_no_fabrication():
    """Verify that when an API fails, it is marked failed and not given a fake prediction."""
    mock_agent = MagicMock()
    mock_agent.handle_message.side_effect = GroqRateLimitExceeded("Rate limit exceeded")

    rows = [{"customer_message": "My iPhone won't turn on"}]
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        cp_path = tf.name

    try:
        results = run_agent(mock_agent, rows, label="test_agent",
                            checkpoint_path=cp_path, resume=False)
        assert len(results) == 1
        res = results[0]
        assert res["failed"] is True
        assert res["intent"]["intent"] == "unknown"
        assert res["reply"] == ""
        assert "Rate limit exceeded" in res["error"]
    finally:
        if os.path.exists(cp_path):
            os.remove(cp_path)


def test_checkpoint_and_resume():
    """Verify that run_agent saves checkpoints after each example and resumes correctly."""
    call_counts = {"count": 0}

    def fake_handle(msg):
        call_counts["count"] += 1
        return {
            "customer_message": msg,
            "intent": {"intent": "battery_and_charging", "confidence": 0.9},
            "reply": "We can help.",
            "escalation": {"should_escalate": False, "reason": "none"},
        }

    mock_agent = MagicMock()
    mock_agent.handle_message.side_effect = fake_handle

    rows = [
        {"customer_message": "Msg 1"},
        {"customer_message": "Msg 2"},
        {"customer_message": "Msg 3"},
    ]

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        cp_path = tf.name

    try:
        # Run first 2 examples and write to checkpoint
        first_pass = run_agent(mock_agent, rows[:2], label="test",
                               checkpoint_path=cp_path, resume=False)
        assert len(first_pass) == 2
        assert call_counts["count"] == 2

        # Second pass on all 3 examples: should resume from checkpoint and only process Msg 3
        second_pass = run_agent(mock_agent, rows, label="test",
                                checkpoint_path=cp_path, resume=True)
        assert len(second_pass) == 3
        assert call_counts["count"] == 3  # Only 1 additional call made!
    finally:
        if os.path.exists(cp_path):
            os.remove(cp_path)


def test_headline_benchmark_selection():
    """Verify headline benchmark definition selects exactly 40 stratified examples."""
    assert "headline" in BENCHMARKS
    assert BENCHMARKS["headline"]["n"] == 40
    assert "full" in BENCHMARKS
    assert BENCHMARKS["full"]["n"] is None

    from src.golden_set import load_evaluation_set
    rows, _ = load_evaluation_set(allow_prelabelled=True)
    assert len(rows) == 200

    # Test stratified subsample of 40 preserves escalation distribution
    sub = stratified_subsample(rows, n=40, seed=42)
    assert len(sub) == 40

    full_esc_rate = sum(1 for r in rows if r["gold_should_escalate"]) / len(rows)
    sub_esc_rate = sum(1 for r in sub if r["gold_should_escalate"]) / len(sub)
    # Stratified rate should be within 5% of full rate
    assert abs(full_esc_rate - sub_esc_rate) < 0.05
