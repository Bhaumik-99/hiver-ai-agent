"""
Escalation rule tests, including adversarial cases.

The deterministic rules are the safety floor of the system, so they are tested
directly rather than only through end-to-end metrics. Note what these tests do
and do not establish: they show the rules fire on the listed cases, which is a
claim about THIS test set, not a guarantee that the agent escalates every unsafe
message in the wild.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.escalation import (
    ESCALATION_KEYWORDS, PROFANITY_PATTERN, RuleOnlyEscalation,
    TrivialEscalation, check_hard_rules,
)

# (message, expected rule category) — every category the deterministic layer owns.
MUST_ESCALATE = [
    ("@AppleSupport my lawyer will be in touch about this", "legal"),
    ("@AppleSupport I am going to sue Apple over this", "legal"),
    ("@AppleSupport my iPhone caught fire while charging", "safety"),
    ("@AppleSupport the battery is swelling and pushing the screen out", "safety"),
    ("@AppleSupport my account was hacked, someone changed my password", "account_security"),
    ("@AppleSupport there is an unauthorized charge on my card", "account_security"),
    ("@AppleSupport I want a refund for this app", "billing_dispute"),
    ("@AppleSupport you charged twice for the same order", "billing_dispute"),
    ("@AppleSupport let me speak to a human please", "explicit_human_request"),
    ("@AppleSupport get me a manager right now", "explicit_human_request"),
    ("@AppleSupport this is the worst experience I have ever had", "severe_frustration"),
    ("@AppleSupport I am going to the media with this", "media_threat"),
    ("@AppleSupport this is fucking broken again", "profanity"),
    ("@AppleSupport call me on 555-123-4567", "pii_detected"),
    ("@AppleSupport my email is someone@example.com, help", "pii_detected"),
]

# Ordinary support traffic that must NOT trip a hard rule. Every one of these is
# a real false positive the earlier keyword list produced.
MUST_NOT_ESCALATE = [
    "@AppleSupport any news on the iOS 11 update?",
    "@AppleSupport damn this update is slow",
    "@AppleSupport that's a crap workaround but ok",
    "@AppleSupport my eyes hurt reading this new font",
    "@AppleSupport the new firework emoji is great",
    "@AppleSupport how do I court a new Apple Watch band?",
    "@AppleSupport wifi keeps dropping, any fix?",
    "@AppleSupport how do I update to the latest version?",
    "@AppleSupport is the store open on Sunday?",
    "@AppleSupport thanks, that worked!",
]


@pytest.mark.parametrize("message,expected_rule", MUST_ESCALATE)
def test_hard_rules_escalate(message, expected_rule):
    result = check_hard_rules(message)
    assert result is not None, f"no rule fired for: {message}"
    assert result["should_escalate"] is True
    assert result["rule"] == expected_rule, (
        f"expected rule {expected_rule}, got {result['rule']} for: {message}")
    assert result["reason"], "an escalation must state a reason"


@pytest.mark.parametrize("message", MUST_NOT_ESCALATE)
def test_hard_rules_do_not_fire_on_ordinary_traffic(message):
    result = check_hard_rules(message)
    assert result is None, (
        f"false positive: {message!r} triggered "
        f"{result['rule'] if result else None}")


def test_safety_recall_on_this_subset_is_total():
    """
    Evidence-bounded claim: the rules fire on every case in MUST_ESCALATE.

    This is 100% recall on a 15-example hand-built safety subset. It is not a
    guarantee of safety on unseen traffic and must not be quoted as one.
    """
    missed = [m for m, _ in MUST_ESCALATE if check_hard_rules(m) is None]
    assert not missed, f"missed {len(missed)} safety cases: {missed}"


def test_profanity_pattern_ignores_mild_expletives():
    assert PROFANITY_PATTERN.search("this is fucking broken")
    assert PROFANITY_PATTERN.search("what a shitshow")
    assert not PROFANITY_PATTERN.search("damn, that's annoying")
    assert not PROFANITY_PATTERN.search("what a crap update")


def test_no_bare_high_frequency_keywords():
    """
    Guard against regressions: single common words as escalation triggers caused
    the original false positives ('news' matching 'any news on the update?').
    """
    too_common = {"news", "fire", "burn", "shock", "hurt", "sue", "court",
                  "damn", "crap", "wtf", "viral", "stolen"}
    offenders = []
    for category, keywords in ESCALATION_KEYWORDS.items():
        for kw in keywords:
            if kw in too_common:
                offenders.append((category, kw))
    assert not offenders, f"bare high-frequency triggers reintroduced: {offenders}"


def test_severe_ambiguity_is_not_a_deterministic_rule():
    """
    Documents a real gap. Very short ambiguous messages ('it did it again') are
    escalation-worthy but are not reliably detectable by keywords, so they are
    delegated to the LLM layer. Asserting that here keeps the report honest about
    which categories have a deterministic guarantee and which do not.
    """
    assert check_hard_rules("@AppleSupport see, it did it again!") is None
    assert check_hard_rules("@AppleSupport still broken") is None


def test_trivial_escalation_never_escalates():
    e = TrivialEscalation()
    for msg, _ in MUST_ESCALATE:
        assert e.decide(msg)["should_escalate"] is False


def test_rule_only_baseline_matches_hard_rules():
    e = RuleOnlyEscalation()
    for msg, rule in MUST_ESCALATE:
        assert e.decide(msg)["should_escalate"] is True
    for msg in MUST_NOT_ESCALATE:
        assert e.decide(msg)["should_escalate"] is False


def test_every_escalation_states_a_reason():
    """A handoff with no stated reason is unusable by the human receiving it."""
    for msg, _ in MUST_ESCALATE:
        r = check_hard_rules(msg)
        assert r["reason"].strip(), f"empty reason for {msg!r}"
        assert 0.0 <= r["confidence"] <= 1.0
