"""
Leakage-invariant tests.

Each test builds a deliberately leaked scenario and asserts the check RAISES.
A leakage guard that never fires is indistinguishable from no guard at all, so
the negative cases matter more than the positive ones here.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.leakage_checks import (
    LeakageError, assert_baseline_out_of_fold, assert_corpus_excludes_golden,
    assert_judge_prompt_clean, assert_no_gold_labels_in_inference,
    assert_retrieval_excludes_self, assert_same_examples, run_all_checks,
)

GOLD = [
    {"example_id": "1", "customer_message": "my battery drains fast",
     "brand_reply": "@x We can help with battery life, DM us: https://t.co/abc"},
    {"example_id": "2", "customer_message": "wifi keeps dropping",
     "brand_reply": "@y Let's look at your network settings, DM us."},
]
CLEAN_CORPUS = [
    {"customer_message": "screen is cracked", "brand_reply": "@z Visit a store."},
    {"customer_message": "how do I update ios", "brand_reply": "@w Settings > General."},
]


def _result(msg, reply="a drafted reply", neighbours=None):
    return {"customer_message": msg, "reply": reply,
            "similar_conversations": neighbours or []}


# ── corpus exclusion ──────────────────────────────────────────────────

def test_clean_corpus_passes():
    rec = assert_corpus_excludes_golden(CLEAN_CORPUS, GOLD)
    assert rec["passed"] is True


def test_corpus_containing_golden_message_raises():
    leaked = CLEAN_CORPUS + [{"customer_message": "my battery drains fast",
                              "brand_reply": "the reference reply"}]
    with pytest.raises(LeakageError, match="golden-set messages"):
        assert_corpus_excludes_golden(leaked, GOLD)


def test_corpus_check_ignores_surrounding_whitespace():
    leaked = CLEAN_CORPUS + [{"customer_message": "  my battery drains fast  ",
                              "brand_reply": "ref"}]
    with pytest.raises(LeakageError):
        assert_corpus_excludes_golden(leaked, GOLD)


# ── retrieval self-exclusion ──────────────────────────────────────────

def test_retrieval_not_returning_self_passes():
    runs = [_result(g["customer_message"],
                    neighbours=[{"customer_message": "unrelated",
                                 "brand_reply": "unrelated reply"}])
            for g in GOLD]
    assert assert_retrieval_excludes_self(runs, GOLD)["passed"] is True


def test_retrieval_returning_own_message_raises():
    runs = [_result(GOLD[0]["customer_message"],
                    neighbours=[{"customer_message": "my battery drains fast",
                                 "brand_reply": "whatever"}]),
            _result(GOLD[1]["customer_message"])]
    with pytest.raises(LeakageError, match="retrieved themselves"):
        assert_retrieval_excludes_self(runs, GOLD)


def test_retrieval_returning_own_reference_reply_raises():
    runs = [_result(GOLD[0]["customer_message"],
                    neighbours=[{"customer_message": "different question",
                                 "brand_reply": GOLD[0]["brand_reply"]}]),
            _result(GOLD[1]["customer_message"])]
    with pytest.raises(LeakageError):
        assert_retrieval_excludes_self(runs, GOLD)


# ── baseline cross-fitting ────────────────────────────────────────────

def test_complete_out_of_fold_coverage_passes():
    ov = {g["customer_message"]: {"intent": "x"} for g in GOLD}
    assert assert_baseline_out_of_fold(ov, GOLD, "simple")["passed"] is True


def test_missing_out_of_fold_prediction_raises():
    ov = {GOLD[0]["customer_message"]: {"intent": "x"}}
    with pytest.raises(LeakageError, match="out-of-fold"):
        assert_baseline_out_of_fold(ov, GOLD, "simple")


# ── judge prompt hygiene ──────────────────────────────────────────────

def test_clean_judge_prompt_passes():
    prompt = 'CUSTOMER MESSAGE: "my battery drains fast"\nAI REPLY: "try this"'
    assert assert_judge_prompt_clean(prompt, GOLD[0]["brand_reply"])["passed"] is True


def test_judge_prompt_containing_reference_raises():
    prompt = f'CUSTOMER: "x"\nReference: "{GOLD[0]["brand_reply"]}"'
    with pytest.raises(LeakageError, match="reference reply"):
        assert_judge_prompt_clean(prompt, GOLD[0]["brand_reply"])


def test_judge_prompt_containing_gold_intent_raises():
    prompt = 'CUSTOMER: "x"\nintent is "battery_and_charging"'
    with pytest.raises(LeakageError, match="gold intent"):
        assert_judge_prompt_clean(prompt, "", gold_intent="battery_and_charging")


# ── reference echo ────────────────────────────────────────────────────

def test_reply_identical_to_reference_raises():
    runs = [_result(GOLD[0]["customer_message"], reply=GOLD[0]["brand_reply"]),
            _result(GOLD[1]["customer_message"])]
    with pytest.raises(LeakageError, match="byte-identical"):
        assert_no_gold_labels_in_inference(runs, GOLD)


def test_distinct_replies_pass():
    runs = [_result(g["customer_message"], reply="a genuinely different reply")
            for g in GOLD]
    assert assert_no_gold_labels_in_inference(runs, GOLD)["passed"] is True


# ── same-examples invariant ───────────────────────────────────────────

def test_same_examples_passes():
    runs = {name: [_result(g["customer_message"]) for g in GOLD]
            for name in ("trivial", "simple", "main")}
    assert assert_same_examples(runs)["passed"] is True


def test_different_examples_across_systems_raises():
    runs = {
        "trivial": [_result(g["customer_message"]) for g in GOLD],
        "main": [_result("a completely different message")],
    }
    with pytest.raises(LeakageError, match="different examples"):
        assert_same_examples(runs)


def test_run_all_checks_on_clean_setup():
    runs = {name: [_result(g["customer_message"], reply=f"{name} reply")
                   for g in GOLD]
            for name in ("trivial", "simple", "main")}
    overrides = {n: {g["customer_message"]: {"intent": "x"} for g in GOLD}
                 for n in ("trivial", "simple")}
    records = run_all_checks(CLEAN_CORPUS, GOLD, runs, overrides)
    assert all(r["passed"] for r in records)
    assert len(records) >= 8
