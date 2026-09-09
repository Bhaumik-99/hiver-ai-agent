"""
Merge the completed annotator files into data/golden_set_final.csv.

Only rows a human marked `human_reviewed` are written. Where both annotators
covered a row and disagree, the row is reported as needing adjudication and
excluded unless --prefer is given, because silently picking one annotator's label
would hide a real disagreement from the agreement statistics.

Usage:
    python scripts/finalize_golden_set.py
    python scripts/finalize_golden_set.py --prefer annotator_a   # adjudicate ties
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.golden_set import (
    DATA_DIR, FINAL_PATH, PRELABELLED_PATH, STATUS_HUMAN,
    annotation_summary, read_rows, validate_schema, write_rows,
)

ANNOTATION_DIR = os.path.join(DATA_DIR, "annotation")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefer", choices=["annotator_a", "annotator_b"],
                    help="which annotator wins when the two disagree")
    args = ap.parse_args()

    files = {}
    for name in ("annotator_a", "annotator_b"):
        p = os.path.join(ANNOTATION_DIR, f"{name}.csv")
        if os.path.exists(p):
            files[name] = {r["example_id"]: r for r in read_rows(p)
                           if r.get("annotation_status") == STATUS_HUMAN}
            print(f"{name}: {len(files[name])} human-reviewed rows")
        else:
            print(f"{name}: missing")

    if not files:
        print("\nNo annotation files found. Run scripts/build_annotation_task.py first.")
        return 1

    total_expected = len(read_rows(PRELABELLED_PATH))

    merged, conflicts = {}, []
    for name, rows in files.items():
        for eid, row in rows.items():
            if eid not in merged:
                merged[eid] = row
                continue
            other = merged[eid]
            same = (other.get("gold_intent") == row.get("gold_intent")
                    and bool(other["gold_should_escalate"]) == bool(row["gold_should_escalate"]))
            if same:
                continue
            conflicts.append({
                "example_id": eid,
                "a_intent": other.get("gold_intent"),
                "b_intent": row.get("gold_intent"),
                "a_escalate": other["gold_should_escalate"],
                "b_escalate": row["gold_should_escalate"],
            })
            if args.prefer == name:
                merged[eid] = row

    if conflicts and not args.prefer:
        print(f"\n{len(conflicts)} rows have unresolved disagreement between annotators:")
        for c in conflicts[:10]:
            print(f"  id={c['example_id']}: intent {c['a_intent']} vs {c['b_intent']}, "
                  f"escalate {c['a_escalate']} vs {c['b_escalate']}")
        print("\nAdjudicate these by hand, or re-run with --prefer annotator_a|annotator_b.")
        print("They are EXCLUDED from the final set for now so the disagreement stays visible.")
        for c in conflicts:
            merged.pop(c["example_id"], None)

    final_rows = sorted(merged.values(), key=lambda r: int(r["example_id"]))

    if not final_rows:
        print("\nNo human-reviewed rows to write. golden_set_final.csv NOT created.")
        print("The evaluation harness will keep refusing to treat pre-labels as a benchmark.")
        return 1

    problems = validate_schema(final_rows)
    if problems:
        print(f"\nSchema problems ({len(problems)}):")
        for p in problems[:10]:
            print(f"  - {p}")
        print("\nFix these before the file can be used. Not writing.")
        return 1

    write_rows(final_rows, FINAL_PATH)
    s = annotation_summary(final_rows)
    print(f"\nWrote {len(final_rows)} human-reviewed rows -> "
          f"{os.path.relpath(FINAL_PATH, DATA_DIR)}")
    print(f"Coverage: {len(final_rows)}/{total_expected} of the sampled golden set")

    if len(final_rows) < 150:
        print(f"\nWARNING: the assignment asks for 150-250 hand-labelled examples "
              f"and this file has {len(final_rows)}. Annotate more rows before "
              f"presenting it as the golden set.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
