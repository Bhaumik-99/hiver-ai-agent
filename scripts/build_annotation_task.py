"""
Build the human annotation task from the heuristic pre-labels.

Produces one CSV per annotator under data/annotation/. Each row carries the
pre-label as a starting point, and the annotator's job is to confirm or correct
it and then set annotation_status=human_reviewed.

Two annotator files are produced so that inter-annotator agreement (Cohen's
kappa) can be computed on a real overlap. The overlap must be annotated
INDEPENDENTLY — if annotator B just reads annotator A's file, kappa measures
nothing.

Usage:
    python scripts/build_annotation_task.py                  # default 50-row overlap
    python scripts/build_annotation_task.py --overlap 60
"""
import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.golden_set import (
    DATA_DIR, PRELABELLED_PATH, SCHEMA, STATUS_PENDING,
    read_rows, write_rows,
)

ANNOTATION_DIR = os.path.join(DATA_DIR, "annotation")
SEED = 42

INSTRUCTIONS = """\
HUMAN ANNOTATION TASK - AppleSupport golden evaluation set
===========================================================

You are producing the ground truth this project is evaluated against. Every row
currently holds a HEURISTIC PRE-LABEL produced by a regex script. Your job is to
read the customer message and confirm or correct it.

For each row:

1. gold_intent - choose exactly one intent from data/intent_taxonomy.json.
   Judge by the customer's PRIMARY need. If a message vents anger AND names a
   technical cause ("iOS 11 destroyed my battery"), label the technical intent
   (battery_and_charging) - the frustration is captured by the escalation flag.
   Use general_inquiry_features only when no other class genuinely fits; it is
   the catch-all and the pre-labeller over-uses it.

2. gold_should_escalate - true when a human agent must take this over.
   Escalate for: explicit request for a person; legal threats; account
   compromise or credential exposure; billing disputes and refunds; physical
   safety (fire, swelling battery, burns); PII posted publicly; abuse; or a
   message so ambiguous that any automated reply would be a guess.
   Do NOT escalate merely because the customer is annoyed.

3. gold_escalation_reason - short snake_case reason when escalating, else blank.
   e.g. legal_threat, account_security, billing_dispute, safety_hazard,
   explicit_human_request, ambiguity_clarification, abuse.

4. gold_reply_quality - 1-5, judging the BRAND'S OWN historical reply in
   brand_reply (not any model output):
     1 no useful content   2 acknowledges only   3 generic deflection to DM
     4 specific actionable step   5 specific step plus correct resource/link

5. labelling_notes - anything ambiguous, and WHY you decided as you did.

6. annotation_status - set to exactly: human_reviewed
   Rows left as pending_review or pre_labelled_heuristic are excluded from the
   final evaluation set.

Do not consult the other annotator's file while working. The overlap rows exist
to measure genuine independent agreement.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--overlap", type=int, default=50,
                    help="rows annotated by BOTH annotators, for kappa")
    args = ap.parse_args()

    rows = read_rows(PRELABELLED_PATH)
    print(f"Loaded {len(rows)} pre-labelled rows")

    os.makedirs(ANNOTATION_DIR, exist_ok=True)

    rng = random.Random(SEED)
    ordered = sorted(rows, key=lambda r: int(r["example_id"]))
    overlap_n = min(args.overlap, len(ordered))
    overlap_ids = set(r["example_id"] for r in rng.sample(ordered, overlap_n))

    overlap = [r for r in ordered if r["example_id"] in overlap_ids]
    rest = [r for r in ordered if r["example_id"] not in overlap_ids]

    # Split the non-overlap remainder between the two annotators, so together
    # they cover all 200 rows exactly once plus the shared overlap.
    half = len(rest) // 2
    a_rows = overlap + rest[:half]
    b_rows = overlap + rest[half:]

    for name, subset in (("annotator_a", a_rows), ("annotator_b", b_rows)):
        prepared = []
        for r in subset:
            row = dict(r)
            # Reset status: a pre-label that still says "pre_labelled_heuristic"
            # must not be mistaken for reviewed work later.
            row["annotation_status"] = STATUS_PENDING
            prepared.append(row)
        prepared.sort(key=lambda r: int(r["example_id"]))
        path = os.path.join(ANNOTATION_DIR, f"{name}.csv")
        if os.path.exists(path):
            print(f"  SKIP {name}.csv (already exists - not overwriting your work)")
            continue
        write_rows(prepared, path)
        print(f"  Wrote {len(prepared):3d} rows -> data/annotation/{name}.csv "
              f"({len(overlap)} shared for kappa)")

    readme = os.path.join(ANNOTATION_DIR, "INSTRUCTIONS.txt")
    with open(readme, "w", encoding="utf-8") as f:
        f.write(INSTRUCTIONS)
        f.write(f"\nOverlap rows (annotated by both, used for Cohen's kappa): "
                f"{overlap_n}\n")
        f.write(f"Overlap example_ids: "
                f"{sorted(int(i) for i in overlap_ids)}\n")
    print(f"  Wrote instructions -> data/annotation/INSTRUCTIONS.txt")

    print("\nNext:")
    print("  1. Annotate both files (set annotation_status=human_reviewed per row)")
    print("  2. python scripts/compute_annotation_agreement.py   # real Cohen's kappa")
    print("  3. python scripts/finalize_golden_set.py            # -> golden_set_final.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
