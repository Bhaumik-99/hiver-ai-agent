"""
Measure how well the LLM judge agrees with a human on reply quality.

Reads the human-scored calibration file, re-runs (or reuses) the LLM judge on the
exact same replies, and reports per-dimension:

  Spearman rank correlation, exact agreement, adjacent (+/-1) agreement,
  mean absolute error, sample count, and quadratic-weighted kappa.

If no human ratings exist the script writes a PENDING record and exits non-zero.
It never substitutes machine labels for human ones: the previous version of this
calibration compared the judge against a regex-generated `gold_reply_quality`
column and reported it as human agreement, which it is not.

Usage:
    python scripts/compute_judge_calibration.py
    python scripts/compute_judge_calibration.py --rejudge   # ignore cached judge scores
"""
import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.llm_judge import RUBRIC, judge_reply, compute_judge_human_agreement, JUDGE_GROQ_MODEL
from src.ollama_client import check_groq

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CALIB_DIR = os.path.join(BASE, "data", "judge_calibration")
HUMAN_PATH = os.path.join(CALIB_DIR, "judge_calibration_human.csv")
TASK_PATH = os.path.join(CALIB_DIR, "judge_calibration_task.csv")
CACHE_PATH = os.path.join(CALIB_DIR, "judge_scores_cache.json")
RESULTS_DIR = os.path.join(BASE, "results")
OUT_PATH = os.path.join(RESULTS_DIR, "judge_calibration.json")

DIMENSIONS = list(RUBRIC.keys())


def quadratic_weighted_kappa(a: list[int], b: list[int],
                             min_rating: int = 1, max_rating: int = 5) -> float:
    """
    Quadratic-weighted kappa for ordinal 1-5 ratings.

    Unweighted kappa treats a 4-vs-5 disagreement as badly as 1-vs-5, which is
    wrong for a rating scale. The quadratic weighting penalises by squared
    distance, so this is the right agreement statistic for the rubric.
    """
    a, b = list(a), list(b)
    n_ratings = max_rating - min_rating + 1
    n = len(a)
    if n == 0:
        return float("nan")

    O = np.zeros((n_ratings, n_ratings))
    for x, y in zip(a, b):
        O[int(x) - min_rating][int(y) - min_rating] += 1

    W = np.zeros((n_ratings, n_ratings))
    for i in range(n_ratings):
        for j in range(n_ratings):
            W[i][j] = ((i - j) ** 2) / ((n_ratings - 1) ** 2)

    hist_a = np.bincount([int(x) - min_rating for x in a], minlength=n_ratings)
    hist_b = np.bincount([int(x) - min_rating for x in b], minlength=n_ratings)
    E = np.outer(hist_a, hist_b).astype(float)
    E = E / E.sum() * O.sum()

    denom = (W * E).sum()
    if denom == 0:
        return float("nan")
    return float(1.0 - (W * O).sum() / denom)


def _write_pending(reason: str, extra: dict) -> int:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    payload = {
        "status": "PENDING",
        "requirement": ("The assignment requires evidence of how well the LLM "
                        "judge agrees with a HUMAN."),
        "reason": reason,
        "how_to_complete": [
            "python scripts/run_evaluation.py --benchmark headline",
            "python scripts/build_judge_calibration_task.py",
            "score every row of data/judge_calibration/judge_calibration_task.csv "
            "and save it as judge_calibration_human.csv",
            "python scripts/compute_judge_calibration.py",
        ],
        **extra,
    }
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\n  STATUS: PENDING - {reason}")
    print(f"  Wrote {os.path.relpath(OUT_PATH, BASE)}")
    print("  No agreement statistic is being estimated in its place.")
    return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rejudge", action="store_true",
                    help="ignore cached judge scores and call the judge again")
    args = ap.parse_args()

    print("=" * 62)
    print("LLM judge vs human calibration")
    print("=" * 62)

    if not os.path.exists(HUMAN_PATH):
        return _write_pending(
            f"no human ratings file at "
            f"{os.path.relpath(HUMAN_PATH, BASE).replace(chr(92), '/')}",
            {"task_file_exists": os.path.exists(TASK_PATH)},
        )

    with open(HUMAN_PATH, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # A row counts only if the human filled every dimension with a valid 1-5.
    scored = []
    for r in rows:
        try:
            human = {d: int(r[f"human_{d}"]) for d in DIMENSIONS}
        except (KeyError, ValueError, TypeError):
            continue
        if not all(1 <= v <= 5 for v in human.values()):
            continue
        try:
            human_overall = float(r.get("human_overall") or
                                  sum(human.values()) / len(human))
        except ValueError:
            human_overall = sum(human.values()) / len(human)
        scored.append({
            "calibration_id": r.get("calibration_id"),
            "system": r.get("system", ""),
            "customer_message": r["customer_message"],
            "generated_reply": r["generated_reply"],
            "human": human,
            "human_overall": human_overall,
        })

    print(f"  Human-scored rows: {len(scored)}/{len(rows)}")

    if len(scored) < 10:
        return _write_pending(
            f"only {len(scored)} fully human-scored rows; too few for a "
            f"meaningful correlation (need >= 10)",
            {"human_scored_rows": len(scored), "total_rows": len(rows)},
        )

    # Judge the same replies.
    cache = {}
    if os.path.exists(CACHE_PATH) and not args.rejudge:
        with open(CACHE_PATH, encoding="utf-8") as f:
            cache = json.load(f)

    use_groq = check_groq()
    print(f"  Judge model: {JUDGE_GROQ_MODEL if use_groq else 'ollama local'}")

    for i, item in enumerate(scored, 1):
        key = str(item["calibration_id"])
        if key in cache:
            item["judge"] = cache[key]
            continue
        print(f"  Judging calibration row {i}/{len(scored)}...", end="\r", flush=True)
        j = judge_reply(item["customer_message"], item["generated_reply"],
                        style_exemplars=None, use_groq=use_groq)
        item["judge"] = {
            d: (j.get(d) or {}).get("score", None) for d in DIMENSIONS
        }
        item["judge"]["overall"] = j.get("overall_score")
        cache[key] = item["judge"]
        with open(CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)
    print(f"  Judged {len(scored)} calibration rows.            ")

    usable = [it for it in scored
              if all(it["judge"].get(d) is not None for d in DIMENSIONS)]
    if len(usable) < 10:
        return _write_pending(
            f"the judge returned usable scores for only {len(usable)} rows",
            {"human_scored_rows": len(scored), "judge_scored_rows": len(usable)},
        )

    per_dim = {}
    for d in DIMENSIONS:
        h = [it["human"][d] for it in usable]
        j = [int(it["judge"][d]) for it in usable]
        agree = compute_judge_human_agreement(j, h)
        qwk = quadratic_weighted_kappa(h, j)
        agree["quadratic_weighted_kappa"] = None if qwk != qwk else round(qwk, 4)
        per_dim[d] = agree

    h_all = [it["human_overall"] for it in usable]
    j_all = [float(it["judge"]["overall"]) for it in usable
             if it["judge"].get("overall") is not None]
    overall = compute_judge_human_agreement(j_all, h_all) if len(j_all) == len(h_all) else {}

    result = {
        "status": "COMPUTED",
        "n_samples": len(usable),
        "judge_model": JUDGE_GROQ_MODEL if use_groq else "ollama local",
        "human_rater_count": 1,
        "per_dimension": per_dim,
        "overall": overall,
        "systems_represented": sorted({it["system"] for it in usable}),
        "caveats": [
            "Single human rater, so this measures agreement with one person's "
            "judgement, not with a consensus of annotators.",
            f"n={len(usable)} is small; the correlations carry wide uncertainty.",
        ],
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"\n  n = {len(usable)} replies, {len(result['systems_represented'])} systems")
    print(f"  {'dimension':<14}{'spearman':>10}{'exact':>9}{'adj':>8}{'MAE':>8}{'QWK':>8}")
    for d, m in per_dim.items():
        qwk = m["quadratic_weighted_kappa"]
        print(f"  {d:<14}{m['spearman_correlation']:>10.3f}"
              f"{m['exact_agreement']:>9.1%}{m['adjacent_agreement']:>8.1%}"
              f"{m['mean_absolute_error']:>8.2f}"
              f"{'  n/a' if qwk is None else f'{qwk:>8.3f}'}")
    if overall:
        print(f"  {'OVERALL':<14}{overall['spearman_correlation']:>10.3f}"
              f"{overall['exact_agreement']:>9.1%}{overall['adjacent_agreement']:>8.1%}"
              f"{overall['mean_absolute_error']:>8.2f}")

    print(f"\n  Wrote {os.path.relpath(OUT_PATH, BASE)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
