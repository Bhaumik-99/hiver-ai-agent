"""
Automated evaluation metrics for the support agent.
Covers: intent classification, reply quality, and escalation decisions.
"""

import json
import os
import numpy as np
from collections import Counter, defaultdict
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report,
)


def evaluate_intent_classification(predictions: list[str], gold_labels: list[str],
                                   label_set: list[str] = None) -> dict:
    """
    Evaluate intent classification performance.

    label_set: the full taxonomy of valid intents. Passing it explicitly is what
    makes macro-F1 comparable ACROSS systems. If we let sklearn infer the label
    set from `set(gold + pred)`, a system that predicts an extra class no gold
    example uses adds a support-0 class scoring F1=0 into the macro average,
    so a system is penalised simply for using more of the taxonomy and each
    system's macro-F1 is divided by a different denominator.

    Returns:
    - accuracy, macro_f1, weighted_f1
    - per_class metrics
    - confusion matrix
    """
    if label_set is None:
        label_set = sorted(set(gold_labels))
    labels = sorted(label_set)

    # Anything predicted outside the taxonomy would be silently dropped by the
    # metrics below, which would hide failures rather than score them. Surface it.
    off_taxonomy = sorted(set(predictions) - set(labels))

    acc = accuracy_score(gold_labels, predictions)
    macro_f1 = f1_score(gold_labels, predictions, labels=labels,
                        average="macro", zero_division=0)
    weighted_f1 = f1_score(gold_labels, predictions, labels=labels,
                           average="weighted", zero_division=0)

    # Per-class report
    report = classification_report(gold_labels, predictions, labels=labels,
                                   output_dict=True, zero_division=0)

    # Confusion matrix
    cm = confusion_matrix(gold_labels, predictions, labels=labels)

    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "per_class": {
            k: {m: round(v, 4) for m, v in metrics.items()}
            for k, metrics in report.items()
            if k not in ("accuracy", "macro avg", "weighted avg")
        },
        "confusion_matrix": {
            "labels": labels,
            "matrix": cm.tolist(),
        },
        "off_taxonomy_predictions": off_taxonomy,
        "n_off_taxonomy": sum(1 for p in predictions if p not in set(labels)),
    }


def evaluate_escalation(predictions: list[bool], gold_labels: list[bool]) -> dict:
    """Evaluate escalation decision performance."""
    acc = accuracy_score(gold_labels, predictions)
    prec = precision_score(gold_labels, predictions, zero_division=0)
    rec = recall_score(gold_labels, predictions, zero_division=0)
    f1 = f1_score(gold_labels, predictions, zero_division=0)

    tp = sum(1 for p, g in zip(predictions, gold_labels) if p and g)
    fp = sum(1 for p, g in zip(predictions, gold_labels) if p and not g)
    fn = sum(1 for p, g in zip(predictions, gold_labels) if not p and g)
    tn = sum(1 for p, g in zip(predictions, gold_labels) if not p and not g)

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "total_escalated": sum(predictions),
        "total_should_escalate": sum(gold_labels),
    }


def compute_rouge_scores(predictions: list[str], references: list[str]) -> dict:
    """Compute ROUGE scores between predicted and reference replies."""
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

        scores = {"rouge1": [], "rouge2": [], "rougeL": []}
        for pred, ref in zip(predictions, references):
            # An example with no reference cannot be scored at all — skip it.
            if not ref:
                continue
            # An EMPTY PREDICTION is a generation failure and must score 0, not be
            # skipped. Skipping it drops the system's worst outputs out of the
            # denominator, so a system that fails to answer 17% of messages gets
            # a ROUGE average computed only over the 83% it did answer.
            if not pred:
                for key in scores:
                    scores[key].append(0.0)
                continue
            result = scorer.score(ref, pred)
            for key in scores:
                scores[key].append(result[key].fmeasure)

        return {
            key: round(float(np.mean(vals)), 4) if vals else 0.0
            for key, vals in scores.items()
        }
    except ImportError:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "error": "rouge_score not installed"}


def compute_reply_stats(predictions: list[str]) -> dict:
    """Compute basic statistics about generated replies."""
    lengths = [len(r) for r in predictions if r]
    word_counts = [len(r.split()) for r in predictions if r]

    return {
        "avg_length_chars": round(float(np.mean(lengths)), 1) if lengths else 0,
        "avg_length_words": round(float(np.mean(word_counts)), 1) if word_counts else 0,
        "min_length": int(min(lengths)) if lengths else 0,
        "max_length": int(max(lengths)) if lengths else 0,
        "empty_replies": sum(1 for r in predictions if not r or not r.strip()),
    }


def full_evaluation(agent_results: list[dict], golden_set: list[dict],
                    label_set: list[str] = None) -> dict:
    """
    Run full evaluation comparing agent results against golden set.

    agent_results: list of agent.handle_message() outputs
    golden_set: list of labelled examples with gold_intent, gold_should_escalate
    label_set: full intent taxonomy, passed so macro-F1 is comparable across systems
    """
    pred_intents = [r["intent"]["intent"] for r in agent_results]
    gold_intents = [g["gold_intent"] for g in golden_set]
    intent_metrics = evaluate_intent_classification(pred_intents, gold_intents,
                                                    label_set=label_set)

    pred_escalate = [r["escalation"]["should_escalate"] for r in agent_results]
    gold_escalate = [g["gold_should_escalate"] for g in golden_set]
    escalation_metrics = evaluate_escalation(pred_escalate, gold_escalate)

    pred_replies = [r["reply"] for r in agent_results]
    ref_replies = [g.get("brand_reply", "") for g in golden_set]
    rouge_metrics = compute_rouge_scores(pred_replies, ref_replies)

    reply_stats = compute_reply_stats(pred_replies)
    times = [r.get("processing_time", 0) for r in agent_results]

    intent_metrics["accuracy_ci95"] = bootstrap_ci(
        [1.0 if p == g else 0.0 for p, g in zip(pred_intents, gold_intents)]
    )
    escalation_metrics["accuracy_ci95"] = bootstrap_ci(
        [1.0 if p == g else 0.0 for p, g in zip(pred_escalate, gold_escalate)]
    )

    return {
        "intent_classification": intent_metrics,
        "escalation": escalation_metrics,
        "rouge_scores": rouge_metrics,
        "reply_stats": reply_stats,
        "performance": {
            "avg_time_seconds": round(float(np.mean(times)), 2) if times else 0,
            "total_time_seconds": round(float(sum(times)), 2) if times else 0,
            "samples_evaluated": len(agent_results),
        },
    }


def bootstrap_ci(values: list, statistic="mean", n_boot: int = 2000,
                 alpha: float = 0.05, seed: int = 42) -> dict:
    """
    Percentile bootstrap confidence interval for a per-example metric.

    On an evaluation set of this size the difference between two systems is
    routinely smaller than the sampling error of either one, so a headline
    point estimate quoted without an interval is not interpretable. `values`
    must be the PER-EXAMPLE scores (e.g. 1/0 correctness), not a summary.
    """
    import numpy as _np
    if not values:
        return {"point": None, "lo": None, "hi": None, "n": 0}
    rng = _np.random.default_rng(seed)
    arr = _np.asarray(values, dtype=float)
    n = len(arr)
    idx = rng.integers(0, n, size=(n_boot, n))
    stats = arr[idx].mean(axis=1)
    return {
        "point": round(float(arr.mean()), 4),
        "lo": round(float(_np.percentile(stats, 100 * alpha / 2)), 4),
        "hi": round(float(_np.percentile(stats, 100 * (1 - alpha / 2))), 4),
        "n": n,
    }


def save_results(results: dict, path: str):
    """Save evaluation results to JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {path}")


def format_results_table(results: dict, name: str = "System") -> str:
    """Format results as a readable text table."""
    lines = [f"\n{'='*60}", f"Evaluation Results: {name}", f"{'='*60}"]

    ic = results.get("intent_classification", {})
    ci = ic.get("accuracy_ci95") or {}
    ci_txt = (f"   [95% CI {ci['lo']:.1%}-{ci['hi']:.1%}]"
              if ci.get("lo") is not None else "")
    lines.append(f"\n--- Intent Classification ---")
    lines.append(f"  Accuracy:     {ic.get('accuracy', 0):.1%}{ci_txt}")
    lines.append(f"  Macro F1:     {ic.get('macro_f1', 0):.1%}")
    lines.append(f"  Weighted F1:  {ic.get('weighted_f1', 0):.1%}")

    esc = results.get("escalation", {})
    eci = esc.get("accuracy_ci95") or {}
    eci_txt = (f"   [95% CI {eci['lo']:.1%}-{eci['hi']:.1%}]"
               if eci.get("lo") is not None else "")
    lines.append(f"\n--- Escalation ---")
    lines.append(f"  Accuracy:     {esc.get('accuracy', 0):.1%}{eci_txt}")
    lines.append(f"  Precision:    {esc.get('precision', 0):.1%}")
    lines.append(f"  Recall:       {esc.get('recall', 0):.1%}")
    lines.append(f"  F1:           {esc.get('f1', 0):.1%}")

    rouge = results.get("rouge_scores", {})
    lines.append(f"\n--- Reply Quality (ROUGE) ---")
    lines.append(f"  ROUGE-1:      {rouge.get('rouge1', 0):.4f}")
    lines.append(f"  ROUGE-2:      {rouge.get('rouge2', 0):.4f}")
    lines.append(f"  ROUGE-L:      {rouge.get('rougeL', 0):.4f}")

    perf = results.get("performance", {})
    lines.append(f"\n--- Performance ---")
    lines.append(f"  Avg time:     {perf.get('avg_time_seconds', 0):.2f}s")
    lines.append(f"  Samples:      {perf.get('samples_evaluated', 0)}")

    return "\n".join(lines)
