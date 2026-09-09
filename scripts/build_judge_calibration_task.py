"""
Build the human calibration task for the LLM judge.

The assignment requires "evidence of how well your judge agrees with a human".
That evidence can only come from a person scoring the same replies the judge
scored, on the same 1-5 rubric. This script exports those replies with blank
human-score columns.

It samples across ALL THREE systems (trivial / simple / main) so the human sees
the full quality range. A calibration set drawn only from the best system gives
the judge no bad replies to disagree about and inflates apparent agreement.

Usage:
    python scripts/build_judge_calibration_task.py            # 40 replies
    python scripts/build_judge_calibration_task.py --n 60
"""
import argparse
import csv
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm_judge import RUBRIC

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE, "results")
OUT_DIR = os.path.join(BASE, "data", "judge_calibration")
OUT_PATH = os.path.join(OUT_DIR, "judge_calibration_task.csv")
SEED = 42

DIMENSIONS = list(RUBRIC.keys())          # relevance, groundedness, helpfulness, tone, completeness
HUMAN_COLS = [f"human_{d}" for d in DIMENSIONS] + ["human_overall"]

INSTRUCTIONS = """\
LLM-JUDGE HUMAN CALIBRATION TASK
=================================

You are scoring AI-generated customer-support replies so we can measure whether
our LLM judge agrees with a human. Fill the human_* columns with integers 1-5.

Score the REPLY given the CUSTOMER MESSAGE. You are not comparing it to any
reference answer, and you should not try to guess what the model scored.

  relevance     1 off-topic  ... 5 directly addresses this specific problem
  groundedness  1 fabricated/contradicts brand practice ... 5 matches how this
                brand actually replies (concise, empathetic, DM for private data)
  helpfulness   1 no next step ... 5 complete actionable resolution guidance
  tone          1 rude/dismissive ... 5 empathetic, professional, on-brand
  completeness  1 ignores most of the query ... 5 covers every aspect

  human_overall the mean of your five scores, or your own holistic 1-5 if you
                prefer - state which convention you used in the notes file.

IMPORTANT
- Do NOT look at results/*_judgments_raw.json while scoring. If you anchor on
  the judge's scores, the agreement number measures nothing.
- The `system` column is included only for bookkeeping. Try to ignore it; do not
  score a reply higher because it came from the main agent.
- Score every row. Partial rows are dropped from the calculation.

When finished, save as data/judge_calibration/judge_calibration_human.csv and run:
    python scripts/compute_judge_calibration.py
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40, help="replies to score in total")
    args = ap.parse_args()

    systems = {}
    for name in ("trivial", "simple", "main"):
        path = os.path.join(RESULTS_DIR, f"{name}_raw_results.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                systems[name] = json.load(f)

    if not systems:
        print("No agent results found in results/. Run the benchmark first:")
        print("  python scripts/run_evaluation.py --benchmark headline")
        return 1

    print(f"Found results for: {', '.join(f'{k} ({len(v)})' for k, v in systems.items())}")

    rng = random.Random(SEED)
    per_system = max(1, args.n // len(systems))

    rows = []
    for name, results in systems.items():
        pool = [r for r in results if r.get("customer_message")]
        picked = rng.sample(pool, min(per_system, len(pool)))
        for r in picked:
            row = {
                "system": name,
                "customer_message": r["customer_message"],
                "generated_reply": r.get("reply", ""),
            }
            row.update({c: "" for c in HUMAN_COLS})
            row["notes"] = ""
            rows.append(row)

    # Shuffle so the human does not score all of one system in a block, which
    # would let them calibrate to that system rather than to the rubric.
    rng.shuffle(rows)
    for i, r in enumerate(rows, 1):
        r["calibration_id"] = i

    os.makedirs(OUT_DIR, exist_ok=True)
    fields = ["calibration_id", "system", "customer_message", "generated_reply"] \
        + HUMAN_COLS + ["notes"]
    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    with open(os.path.join(OUT_DIR, "INSTRUCTIONS.txt"), "w", encoding="utf-8") as f:
        f.write(INSTRUCTIONS)

    print(f"\nWrote {len(rows)} replies -> data/judge_calibration/judge_calibration_task.csv")
    print(f"  ({per_system} per system, shuffled, seed={SEED})")
    print("Wrote instructions -> data/judge_calibration/INSTRUCTIONS.txt")
    print("\nNext: score every row, save as judge_calibration_human.csv, then run")
    print("  python scripts/compute_judge_calibration.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
