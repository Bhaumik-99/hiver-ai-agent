"""
Evaluation-metric tests.

The macro-F1 label-set test is the one that matters most: without an explicit
label set each system is scored over a different denominator and the headline
comparison is not a comparison at all.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluator import (
    bootstrap_ci, compute_reply_stats, compute_rouge_scores,
    evaluate_escalation, evaluate_intent_classification, full_evaluation,
)

TAXONOMY = ["battery_and_charging", "connectivity_and_network",
            "general_inquiry_features", "software_update_os"]


# ── intent metrics ────────────────────────────────────────────────────

def test_perfect_classification():
    gold = ["battery_and_charging", "software_update_os"]
    m = evaluate_intent_classification(gold, gold, label_set=TAXONOMY)
    assert m["accuracy"] == 1.0
    # macro-F1 averages over the whole taxonomy, so unused classes pull it down.
    assert 0 < m["macro_f1"] <= 1.0
    assert m["weighted_f1"] == 1.0


def test_macro_f1_denominator_is_the_taxonomy_not_the_union():
    """
    Two systems on the same gold labels must be scored over the same classes.
    Previously the label set was set(gold+pred), so a system that used an extra
    class was penalised for it and each system got a different denominator.
    """
    gold = ["battery_and_charging"] * 4
    conservative = ["battery_and_charging"] * 4
    adventurous = ["battery_and_charging"] * 3 + ["connectivity_and_network"]

    a = evaluate_intent_classification(conservative, gold, label_set=TAXONOMY)
    b = evaluate_intent_classification(adventurous, gold, label_set=TAXONOMY)
    assert a["confusion_matrix"]["labels"] == b["confusion_matrix"]["labels"] == sorted(TAXONOMY)
    assert len(a["confusion_matrix"]["matrix"]) == len(TAXONOMY)


def test_off_taxonomy_predictions_are_surfaced_not_hidden():
    gold = ["battery_and_charging", "battery_and_charging"]
    pred = ["battery_and_charging", "hallucinated_intent"]
    m = evaluate_intent_classification(pred, gold, label_set=TAXONOMY)
    assert m["n_off_taxonomy"] == 1
    assert "hallucinated_intent" in m["off_taxonomy_predictions"]
    assert m["accuracy"] == 0.5


def test_confusion_matrix_shape_matches_label_set():
    gold = ["battery_and_charging", "software_update_os"]
    pred = ["battery_and_charging", "battery_and_charging"]
    m = evaluate_intent_classification(pred, gold, label_set=TAXONOMY)
    cm = m["confusion_matrix"]["matrix"]
    assert len(cm) == len(TAXONOMY) and all(len(r) == len(TAXONOMY) for r in cm)


# ── escalation metrics ────────────────────────────────────────────────

def test_escalation_confusion_counts():
    pred = [True, True, False, False]
    gold = [True, False, True, False]
    m = evaluate_escalation(pred, gold)
    assert m["confusion"] == {"tp": 1, "fp": 1, "fn": 1, "tn": 1}
    assert m["precision"] == 0.5
    assert m["recall"] == 0.5
    assert m["f1"] == 0.5
    assert m["accuracy"] == 0.5


def test_never_escalating_baseline_scores_zero_not_undefined():
    m = evaluate_escalation([False] * 4, [True, False, False, False])
    assert m["precision"] == 0.0 and m["recall"] == 0.0 and m["f1"] == 0.0
    assert m["accuracy"] == 0.75          # accuracy alone flatters this baseline
    assert m["confusion"]["fn"] == 1


# ── ROUGE ─────────────────────────────────────────────────────────────

def test_empty_prediction_scores_zero_rather_than_being_skipped():
    """
    Skipping empty predictions silently removed a system's worst outputs from
    the average, so a system failing 50% of messages looked as good as one
    answering them all.
    """
    refs = ["please restart your device", "please restart your device"]
    both = compute_rouge_scores(["please restart your device",
                                 "please restart your device"], refs)
    half = compute_rouge_scores(["please restart your device", ""], refs)
    assert half["rouge1"] == pytest.approx(both["rouge1"] / 2, abs=1e-6)


def test_rouge_skips_examples_with_no_reference():
    scores = compute_rouge_scores(["some reply"], [""])
    assert scores["rouge1"] == 0.0


# ── reply stats ───────────────────────────────────────────────────────

def test_reply_stats_counts_empty():
    s = compute_reply_stats(["hello there", "", "  ", "another reply"])
    assert s["empty_replies"] == 2


# ── bootstrap CI ──────────────────────────────────────────────────────

def test_bootstrap_ci_brackets_the_point_estimate():
    ci = bootstrap_ci([1.0] * 30 + [0.0] * 20)
    assert ci["point"] == pytest.approx(0.6, abs=1e-9)
    assert ci["lo"] < ci["point"] < ci["hi"]
    assert ci["n"] == 50


def test_bootstrap_ci_is_deterministic():
    v = [1.0, 0.0] * 25
    assert bootstrap_ci(v) == bootstrap_ci(v)


def test_bootstrap_ci_widens_on_small_samples():
    small = bootstrap_ci([1.0] * 6 + [0.0] * 4)
    large = bootstrap_ci([1.0] * 600 + [0.0] * 400)
    assert (small["hi"] - small["lo"]) > (large["hi"] - large["lo"])


def test_bootstrap_ci_handles_empty_input():
    assert bootstrap_ci([])["point"] is None


# ── full evaluation ───────────────────────────────────────────────────

def test_full_evaluation_shape():
    results = [
        {"customer_message": "m1", "reply": "r1",
         "intent": {"intent": "battery_and_charging"},
         "escalation": {"should_escalate": False}, "processing_time": 1.0},
        {"customer_message": "m2", "reply": "r2",
         "intent": {"intent": "software_update_os"},
         "escalation": {"should_escalate": True}, "processing_time": 2.0},
    ]
    gold = [
        {"gold_intent": "battery_and_charging", "gold_should_escalate": False,
         "brand_reply": "ref one"},
        {"gold_intent": "software_update_os", "gold_should_escalate": True,
         "brand_reply": "ref two"},
    ]
    m = full_evaluation(results, gold, label_set=TAXONOMY)
    assert m["intent_classification"]["accuracy"] == 1.0
    assert m["escalation"]["f1"] == 1.0
    assert m["performance"]["samples_evaluated"] == 2
    assert m["intent_classification"]["accuracy_ci95"]["n"] == 2
    assert m["escalation"]["accuracy_ci95"]["n"] == 2
    for key in ("rouge1", "rouge2", "rougeL"):
        assert key in m["rouge_scores"]
