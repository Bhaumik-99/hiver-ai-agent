"""
One-off migration of the historical golden-set files into the staged schema.

Before: data/golden_set.csv (unlabelled) and data/golden_set_labelled.csv, where
the second name implied human labels but was in fact the output of the heuristic
in scripts/label_golden_set.py.

After:
  data/golden_set_raw.csv         - the sampled messages, no labels
  data/golden_set_prelabelled.csv - heuristic pre-labels, annotation_status=pre_labelled_heuristic
  data/golden_set_final.csv       - written only by scripts/finalize_golden_set.py,
                                    once a human has reviewed every row

Run once; it is idempotent.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.golden_set import (
    DATA_DIR, RAW_PATH, PRELABELLED_PATH, SCHEMA,
    STATUS_PRELABELLED, read_rows, write_rows, annotation_summary,
)

LEGACY_RAW = os.path.join(DATA_DIR, "golden_set.csv")
LEGACY_LABELLED = os.path.join(DATA_DIR, "golden_set_labelled.csv")


def main():
    if not os.path.exists(LEGACY_LABELLED):
        print(f"Nothing to migrate: {LEGACY_LABELLED} not found.")
        return 0

    rows = read_rows(LEGACY_LABELLED)
    print(f"Read {len(rows)} rows from {os.path.basename(LEGACY_LABELLED)}")

    for i, r in enumerate(rows, 1):
        r.setdefault("example_id", str(i))
        # These labels came from a regex, so they are pre-labels by definition.
        # Recording that in the data itself is the whole point of the migration.
        r["annotation_status"] = STATUS_PRELABELLED
        note = (r.get("labelling_notes") or "").strip()
        marker = "heuristic pre-label from scripts/label_golden_set.py"
        r["labelling_notes"] = f"{note} | {marker}" if note and marker not in note else (note or marker)

    write_rows(rows, PRELABELLED_PATH)
    print(f"Wrote {len(rows)} pre-labelled rows -> {os.path.relpath(PRELABELLED_PATH, DATA_DIR)}")

    # Raw stage: the same messages with every label column blanked.
    raw = []
    for r in rows:
        raw.append({
            "example_id": r["example_id"],
            "customer_message": r["customer_message"],
            "brand_reply": r.get("brand_reply", ""),
            "difficulty": r.get("difficulty", ""),
            "gold_intent": "", "gold_should_escalate": "false",
            "gold_escalation_reason": "", "gold_reply_quality": "",
            "labelling_notes": "", "annotation_status": "pending_review",
        })
    write_rows(raw, RAW_PATH)
    print(f"Wrote {len(raw)} raw rows        -> {os.path.relpath(RAW_PATH, DATA_DIR)}")

    s = annotation_summary(rows)
    print(f"\nAnnotation status: {s['human_reviewed']}/{s['total']} human-reviewed, "
          f"{s['pre_labelled_heuristic']} heuristic pre-labels")
    print("The evaluation harness will refuse to report these as a benchmark "
          "until a human reviews them (see src/golden_set.load_evaluation_set).")

    for legacy in (LEGACY_RAW, LEGACY_LABELLED):
        if os.path.exists(legacy):
            os.remove(legacy)
            print(f"Removed legacy file {os.path.basename(legacy)} "
                  f"(its name asserted a provenance it did not have)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
