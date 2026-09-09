"""Reply-validation tests: the checks that stand between the agent and a live handle."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.reply_validation import (
    TWITTER_CHAR_LIMIT, validate_batch, validate_intent_output, validate_reply,
)

GOOD = "@customer Sorry about that! Try Settings > General > Software Update. If it persists, DM us: https://t.co/GDrqU22YpT ^AL"


def _types(res):
    return {v["type"] for v in res["violations"]}


def test_good_reply_passes():
    res = validate_reply(GOOD)
    assert res["valid"], res["violations"]


def test_empty_reply_flagged():
    for empty in ("", "   ", None):
        res = validate_reply(empty)
        assert not res["valid"]
        assert "empty_reply" in _types(res)


def test_char_limit_enforced():
    assert validate_reply("a" * TWITTER_CHAR_LIMIT)["valid"]
    res = validate_reply("a" * (TWITTER_CHAR_LIMIT + 1))
    assert "char_limit_exceeded" in _types(res)


def test_context_leak_markers_detected():
    for leaked in [
        "@customer Example 1: Customer: my phone broke",
        "@customer Based on the examples, try restarting.",
        "@customer Detected intent: battery_and_charging. Try charging.",
        "@customer As an AI, I suggest restarting your device.",
    ]:
        assert "context_leak" in _types(validate_reply(leaked)), leaked


def test_verbatim_echo_of_retrieved_message_is_a_leak():
    neighbour = {"customer_message": "my iphone screen went completely black after the update",
                 "brand_reply": "@x DM us"}
    reply = ("@customer my iphone screen went completely black after the update "
             "- let's fix that, DM us.")
    assert "context_leak" in _types(validate_reply(reply, [neighbour]))


def test_pii_detected():
    for pii in [
        "@customer email us at someone@example.com",
        "@customer call 555-123-4567 for help",
        "@customer your card 4111 1111 1111 1111 was charged",
    ]:
        assert "pii_exposed" in _types(validate_reply(pii)), pii


def test_invalid_support_url_detected():
    res = validate_reply("@customer see https://apple-support-help.xyz/fix")
    assert "invalid_support_url" in _types(res)


def test_allowed_brand_urls_pass():
    for url in ["https://t.co/GDrqU22YpT",
                "https://support.apple.com/en-us/HT201222",
                "https://iforgot.apple.com"]:
        res = validate_reply(f"@customer try this: {url}")
        assert "invalid_support_url" not in _types(res), url


def test_malformed_tco_link_detected():
    res = validate_reply("@customer here https://t.co/ab")
    assert "invalid_support_url" in _types(res)


def test_intent_validation():
    taxonomy = ["battery_and_charging", "general_inquiry_features"]
    assert validate_intent_output("battery_and_charging", taxonomy)["valid"]
    bad = validate_intent_output("made_up_intent", taxonomy)
    assert not bad["valid"]
    assert bad["violations"][0]["type"] == "invalid_intent"


def test_validate_batch_counts_violations():
    results = [
        {"customer_message": "m1", "reply": GOOD,
         "intent": {"intent": "battery_and_charging"}, "similar_conversations": []},
        {"customer_message": "m2", "reply": "",
         "intent": {"intent": "battery_and_charging"}, "similar_conversations": []},
        {"customer_message": "m3", "reply": "a" * 400,
         "intent": {"intent": "not_a_real_intent"}, "similar_conversations": []},
    ]
    summary = validate_batch(results, ["battery_and_charging"])
    assert summary["n_replies"] == 3
    assert summary["n_replies_with_violation"] == 2
    assert summary["counts_by_type"]["empty_reply"] == 1
    assert summary["counts_by_type"]["char_limit_exceeded"] == 1
    assert summary["counts_by_type"]["invalid_intent"] == 1
    assert summary["n_over_char_limit"] == 1
    assert 0.0 <= summary["violation_rate"] <= 1.0


def test_validate_batch_reports_zero_honestly():
    """A clean batch must report zero, and a dirty one must never report zero."""
    clean = [{"customer_message": "m", "reply": GOOD,
              "intent": {"intent": "battery_and_charging"},
              "similar_conversations": []}]
    assert validate_batch(clean, ["battery_and_charging"])["n_replies_with_violation"] == 0

    dirty = [{"customer_message": "m", "reply": "",
              "intent": {"intent": "battery_and_charging"},
              "similar_conversations": []}]
    assert validate_batch(dirty, ["battery_and_charging"])["n_replies_with_violation"] == 1
