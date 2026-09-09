# Quarantine

## golden_set_final.REJECTED.csv

This file appeared during the audit claiming all 200 rows were `human_reviewed`.
Every row's `gold_*` fields were byte-identical to the heuristic output in
`data/golden_set_prelabelled.csv`, and every row still carried the generator's
own note, `heuristic pre-label from scripts/label_golden_set.py`.

It was the pre-label file with the status column rewritten. No human reviewed it.

Accepting it would have let the harness report a benchmark against regex-generated
labels as if they were hand-labelled ground truth, which is the specific
misrepresentation this project's evaluation is built to prevent.

`src.golden_set.detect_unreviewed_relabelling` now catches this pattern and
`load_evaluation_set` refuses it. See `tests/test_golden_set.py`.

To produce a genuine `data/golden_set_final.csv`:

    python scripts/build_annotation_task.py     # writes data/annotation/
    # review the rows by hand and correct the labels that are wrong
    python scripts/finalize_golden_set.py
