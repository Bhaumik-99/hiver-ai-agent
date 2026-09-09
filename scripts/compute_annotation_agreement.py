"""
Inter-annotator agreement on the overlap rows of the golden set.

Reports Cohen's kappa, raw agreement and the disagreement count for BOTH
labelling decisions (intent and escalation), plus the actual disagreeing rows so
they can be adjudicated.

This script computes a number only when two genuinely independent human
annotations exist. If they do not, it writes a PENDING record and exits non-zero.
It will never emit a kappa derived from heuristic pre-labels: agreement between
a regex and itself is 1.0 and means nothing.

Usage:
    python scripts/compute_annotation_agreement.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.golden_set import DATA_DIR, STATUS_HUMAN, read_rows

ANNOTATION_DIR = os.path.join(DATA_DIR, "annotation")
A_PATH = os.path.join(ANNOTATION_DIR, "annotator_a.csv")
B_PATH = os.path.join(ANNOTATION_DIR, "annotator_b.csv")
RESULTS_DIR = os.path.join(os.path.dirname(DATA_DIR), "results")
OUT_PATH = os.path.join(RESULTS_DIR, "annotation_agreement.json")


def cohens_kappa(a: list, b: list) -> float:
    """
    Unweighted Cohen's kappa for two raters over the same items.

    kappa = (Po - Pe) / (1 - Pe), where Po is observed agreement and Pe is the
    agreement expected from the raters' marginal label frequencies.
    """
    if not a:
        return float("nan")
    n = len(a)
    po = sum(1 for x, y in zip(a, b) if x == y) / n

    labels = set(a) | set(b)
    pe = sum((a.count(l) / n) * (b.count(l) / n) for l in labels)

    if abs(1.0 - pe) < 1e-12:
        # Both raters used a single identical label for everything: kappa is
        # undefined, and reporting 1.0 here would overstate agreement.
        return float("nan")
    return (po - pe) / (1 - pe)


def _human_rows(path: str, who: str) -> dict:
    if not os.path.exists(path):
        print(f"  {who}: MISSING ({os.path.relpath(path, DATA_DIR)})")
        return {}
    rows = read_rows(path)
    human = {r["example_id"]: r for r in rows
             if r.get("annotation_status") == STATUS_HUMAN}
    print(f"  {who}: {len(human)}/{len(rows)} rows marked {STATUS_HUMAN}")
    return human


def main():
    print("=" * 62)
    print("Inter-annotator agreement (human vs human)")
    print("=" * 62)

    a_rows = _human_rows(A_PATH, "annotator_a")
    b_rows = _human_rows(B_PATH, "annotator_b")

    shared = sorted(set(a_rows) & set(b_rows), key=lambda x: int(x))
    print(f"\n  Overlap rows annotated by BOTH: {len(shared)}")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    if len(shared) < 2:
        pending = {
            "status": "PENDING",
            "reason": (
                "Fewer than 2 rows carry independent human annotations from both "
                "annotators, so Cohen's kappa is not computable. This is reported "
                "as PENDING rather than estimated: an agreement statistic that "
                "nobody measured is not a result."
            ),
            "annotator_a_human_rows": len(a_rows),
            "annotator_b_human_rows": len(b_rows),
            "overlap_rows": len(shared),
            "how_to_complete": [
                "python scripts/build_annotation_task.py",
                "annotate data/annotation/annotator_a.csv and annotator_b.csv "
                "independently, setting annotation_status=human_reviewed",
                "python scripts/compute_annotation_agreement.py",
            ],
        }
        with open(OUT_PATH, "w", encoding="utf-8") as f:
            json.dump(pending, f, indent=2)
        print("\n  STATUS: PENDING - no genuine human-human agreement available.")
        print(f"  Wrote {os.path.relpath(OUT_PATH, DATA_DIR)}")
        print("\n  Nothing is being estimated in its place. Any kappa quoted in the")
        print("  report must come from this file once annotation is complete.")
        return 1

    intent_a = [a_rows[i].get("gold_intent", "") for i in shared]
    intent_b = [b_rows[i].get("gold_intent", "") for i in shared]
    esc_a = [bool(a_rows[i]["gold_should_escalate"]) for i in shared]
    esc_b = [bool(b_rows[i]["gold_should_escalate"]) for i in shared]

    def block(name, xa, xb):
        raw = sum(1 for x, y in zip(xa, xb) if x == y) / len(xa)
        k = cohens_kappa(xa, xb)
        disagreements = [
            {"example_id": shared[i], "annotator_a": xa[i], "annotator_b": xb[i],
             "customer_message": a_rows[shared[i]]["customer_message"][:160]}
            for i in range(len(shared)) if xa[i] != xb[i]
        ]
        return {
            "cohens_kappa": None if k != k else round(k, 4),
            "raw_agreement": round(raw, 4),
            "n_items": len(xa),
            "n_disagreements": len(disagreements),
            "disagreements": disagreements[:25],
        }

    result = {
        "status": "COMPUTED",
        "n_overlap_items": len(shared),
        "intent": block("intent", intent_a, intent_b),
        "escalation": block("escalation", esc_a, esc_b),
        "note": ("Computed from two independent human annotation files. "
                 "kappa=null means it is undefined for this data (e.g. both "
                 "raters used a single label), not that agreement was perfect."),
    }

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    for dim in ("intent", "escalation"):
        d = result[dim]
        k = d["cohens_kappa"]
        print(f"\n  {dim.upper()}")
        print(f"    Cohen's kappa:   {'undefined' if k is None else k}")
        print(f"    Raw agreement:   {d['raw_agreement']:.1%}")
        print(f"    Disagreements:   {d['n_disagreements']}/{d['n_items']}")

    print(f"\n  Wrote {os.path.relpath(OUT_PATH, DATA_DIR)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
