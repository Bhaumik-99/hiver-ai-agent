"""
Golden evaluation set: schema, provenance enforcement, and deterministic sampling.

The single most important rule in this file: labels produced by a heuristic or by
an LLM are PRE-LABELS, not ground truth. `scripts/label_golden_set.py` writes
pre-labels to speed a human up; it does not produce a human-labelled set. The
assignment asks for 150-250 hand-labelled examples, so the evaluation harness
refuses to silently treat pre-labels as final, and anything computed from
pre-labels is stamped PROVISIONAL everywhere it surfaces.
"""

import csv
import hashlib
import os
import random

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

# Stage files. Each stage is a separate artifact so provenance is visible in the
# filesystem rather than asserted in prose that can drift from the data.
RAW_PATH = os.path.join(DATA_DIR, "golden_set_raw.csv")
PRELABELLED_PATH = os.path.join(DATA_DIR, "golden_set_prelabelled.csv")
FINAL_PATH = os.path.join(DATA_DIR, "golden_set_final.csv")

SCHEMA = [
    "example_id",
    "customer_message",
    "brand_reply",
    "difficulty",
    "gold_intent",
    "gold_should_escalate",
    "gold_escalation_reason",
    "gold_reply_quality",
    "labelling_notes",
    "annotation_status",
]

# Values permitted in the annotation_status column.
STATUS_PRELABELLED = "pre_labelled_heuristic"   # machine guess, NOT ground truth
STATUS_HUMAN = "human_reviewed"                 # a person read and confirmed/corrected it
STATUS_PENDING = "pending_review"               # awaiting a human
VALID_STATUSES = {STATUS_PRELABELLED, STATUS_HUMAN, STATUS_PENDING}


class GoldenSetProvenanceError(RuntimeError):
    """Raised when the harness is asked to treat pre-labels as human ground truth."""


def _parse_bool(value) -> bool:
    return str(value).strip().lower() in ("true", "1", "yes", "y")


def _parse_quality(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def read_rows(path: str) -> list[dict]:
    """Read a golden-set CSV, normalising the typed columns."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row["gold_should_escalate"] = _parse_bool(row.get("gold_should_escalate"))
            row["gold_reply_quality"] = _parse_quality(row.get("gold_reply_quality"))
            row.setdefault("annotation_status", STATUS_PENDING)
            rows.append(row)
    return rows


def write_rows(rows: list[dict], path: str) -> None:
    """Write rows using the canonical schema and column order."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SCHEMA, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            out = {k: row.get(k, "") for k in SCHEMA}
            out["gold_should_escalate"] = (
                "true" if _parse_bool(row.get("gold_should_escalate")) else "false"
            )
            writer.writerow(out)


def file_hash(path: str) -> str:
    """SHA-256 of a file, recorded in run config so results trace to exact inputs."""
    if not os.path.exists(path):
        return "missing"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def validate_schema(rows: list[dict]) -> list[str]:
    """Return a list of schema violations; empty means the file is well-formed."""
    problems = []
    if not rows:
        return ["golden set is empty"]

    missing = [c for c in SCHEMA if c not in rows[0]]
    if missing:
        problems.append(f"missing columns: {missing}")

    seen_ids = set()
    for i, r in enumerate(rows):
        rid = r.get("example_id")
        if not rid:
            problems.append(f"row {i}: empty example_id")
        elif rid in seen_ids:
            problems.append(f"row {i}: duplicate example_id {rid}")
        seen_ids.add(rid)

        if not (r.get("customer_message") or "").strip():
            problems.append(f"row {i}: empty customer_message")
        if not (r.get("gold_intent") or "").strip():
            problems.append(f"row {i}: empty gold_intent")

        status = r.get("annotation_status", "")
        if status not in VALID_STATUSES:
            problems.append(f"row {i}: invalid annotation_status {status!r}")

        q = r.get("gold_reply_quality")
        if q is not None and not (1 <= q <= 5):
            problems.append(f"row {i}: gold_reply_quality {q} outside 1-5")

    return problems


def annotation_summary(rows: list[dict]) -> dict:
    """Count rows by annotation status."""
    counts = {s: 0 for s in VALID_STATUSES}
    for r in rows:
        counts[r.get("annotation_status", STATUS_PENDING)] = (
            counts.get(r.get("annotation_status", STATUS_PENDING), 0) + 1
        )
    total = len(rows)
    return {
        "total": total,
        "human_reviewed": counts.get(STATUS_HUMAN, 0),
        "pre_labelled_heuristic": counts.get(STATUS_PRELABELLED, 0),
        "pending_review": counts.get(STATUS_PENDING, 0),
        "fully_human_reviewed": counts.get(STATUS_HUMAN, 0) == total and total > 0,
    }


HEURISTIC_MARKER = "heuristic pre-label from scripts/label_golden_set.py"


def detect_unreviewed_relabelling(rows: list[dict]) -> dict:
    """
    Detect rows that CLAIM human review but show no evidence of it.

    Setting annotation_status=human_reviewed is a one-line edit, so the status
    column alone is not evidence that a person looked at anything. This compares
    the claim against the pre-labelled file: if a row is marked human_reviewed
    while every gold_* field is byte-identical to the heuristic output AND its
    notes still carry the generator's marker, the review did not happen.

    A genuine review can of course agree with the pre-label on many rows — what
    is not credible is agreeing on ALL of them while leaving the machine's own
    provenance note in place on every single one.
    """
    if not os.path.exists(PRELABELLED_PATH):
        return {"checked": False, "reason": "no pre-labelled file to compare against"}

    pre = {r["example_id"]: r for r in read_rows(PRELABELLED_PATH)}
    label_fields = ("gold_intent", "gold_should_escalate",
                    "gold_escalation_reason", "gold_reply_quality")

    claimed = [r for r in rows if r.get("annotation_status") == STATUS_HUMAN]
    identical, marker_left = 0, 0
    for r in claimed:
        p = pre.get(r["example_id"])
        if not p:
            continue
        if all(str(r.get(f)) == str(p.get(f)) for f in label_fields):
            identical += 1
        if HEURISTIC_MARKER in (r.get("labelling_notes") or ""):
            marker_left += 1

    n = len(claimed)
    suspicious = bool(n) and identical == n and marker_left == n
    return {
        "checked": True,
        "n_claiming_human_review": n,
        "n_identical_to_prelabels": identical,
        "n_retaining_generator_marker": marker_left,
        "suspicious": suspicious,
    }


def load_evaluation_set(allow_prelabelled: bool = False,
                        path: str = None) -> tuple[list[dict], dict]:
    """
    Load the set the harness will score against.

    Prefers `golden_set_final.csv` (human-reviewed). Falls back to the pre-labelled
    file ONLY when explicitly permitted, and returns provenance saying so. The
    fallback is deliberately loud: silently scoring against machine-generated
    labels while calling the result a benchmark is the exact failure this module
    exists to prevent.

    Returns (rows, provenance).
    """
    if path:
        chosen, source = path, "explicit_path"
    elif os.path.exists(FINAL_PATH):
        chosen, source = FINAL_PATH, "final"
    elif os.path.exists(PRELABELLED_PATH):
        chosen, source = PRELABELLED_PATH, "prelabelled"
    else:
        raise FileNotFoundError(
            f"No golden set found. Expected {FINAL_PATH} (human-reviewed) or "
            f"{PRELABELLED_PATH} (pre-labels). Run scripts/build_golden_set.py "
            f"then scripts/label_golden_set.py."
        )

    rows = read_rows(chosen)
    summary = annotation_summary(rows)
    problems = validate_schema(rows)

    provenance = {
        "path": os.path.relpath(chosen, os.path.dirname(DATA_DIR)).replace("\\", "/"),
        "source_stage": source,
        "sha256_16": file_hash(chosen),
        "schema_problems": problems,
        **summary,
    }

    # A row can be flipped to human_reviewed without anyone reading it. Check the
    # claim against the pre-labels before trusting it.
    relabel = detect_unreviewed_relabelling(rows)
    provenance["relabelling_check"] = relabel
    if relabel.get("suspicious") and not allow_prelabelled:
        raise GoldenSetProvenanceError(
            f"\n{chosen}\nmarks all {relabel['n_claiming_human_review']} rows as "
            f"'{STATUS_HUMAN}', but every one of them is byte-identical to the "
            f"heuristic output in {os.path.basename(PRELABELLED_PATH)} and still "
            f"carries the generator's own provenance note.\n\n"
            "That is the pre-label file with the status column rewritten, not a "
            "human-reviewed golden set. Flipping the status does not make labels "
            "human, and reporting a benchmark against it would misrepresent the "
            "evaluation.\n\n"
            "Either complete the annotation for real:\n"
            "  1. python scripts/build_annotation_task.py\n"
            "  2. review the rows and correct the labels that are wrong\n"
            "  3. python scripts/finalize_golden_set.py\n\n"
            "or run explicitly on pre-labels with --allow-prelabelled, which "
            "stamps every result PROVISIONAL."
        )
    if relabel.get("suspicious"):
        provenance["warning_relabelling"] = (
            "annotation_status says human_reviewed but the labels are identical "
            "to the heuristic pre-labels; treating them as PROVISIONAL."
        )

    if not summary["fully_human_reviewed"] or relabel.get("suspicious"):
        if not allow_prelabelled:
            raise GoldenSetProvenanceError(
                f"\n{chosen}\ncontains {summary['human_reviewed']}/{summary['total']} "
                f"human-reviewed rows.\n"
                "The assignment requires 150-250 HAND-LABELLED examples, so this "
                "harness will not report a benchmark against machine-generated "
                "labels as if it were one.\n\n"
                "To complete the golden set:\n"
                "  1. python scripts/build_annotation_task.py     "
                "# writes data/annotation/ for human review\n"
                "  2. review every row, set annotation_status=human_reviewed\n"
                "  3. python scripts/finalize_golden_set.py       "
                "# writes data/golden_set_final.csv\n\n"
                "To run anyway on pre-labels, pass --allow-prelabelled. Every "
                "result will be stamped PROVISIONAL."
            )
        provenance["labels_are_provisional"] = True
        provenance["warning"] = (
            "PROVISIONAL RESULTS: scored against heuristic pre-labels, not human "
            "ground truth. These numbers measure agreement with "
            "scripts/label_golden_set.py, not with a human annotator."
        )
    else:
        provenance["labels_are_provisional"] = False

    return rows, provenance


def stratified_subsample(rows: list[dict], n: int, seed: int = 42) -> list[dict]:
    """
    Deterministic subsample preserving the escalation base rate and, within it,
    the intent distribution.

    Taking `rows[:n]` instead (the previous behaviour) gave a 10% escalation base
    rate against the full set's 34%, so the escalation metrics described a
    different distribution from the one being claimed.
    """
    if n >= len(rows):
        return sorted(rows, key=lambda r: str(r["example_id"]))

    rng = random.Random(seed)
    # Stratify jointly on escalation flag and intent so both marginals are held.
    buckets: dict[tuple, list] = {}
    for r in rows:
        key = (bool(r["gold_should_escalate"]), r.get("gold_intent", ""))
        buckets.setdefault(key, []).append(r)

    for key in buckets:
        buckets[key].sort(key=lambda r: str(r["example_id"]))

    # Largest-remainder allocation keeps the sample size exact.
    total = len(rows)
    quotas, remainders = {}, []
    for key, members in buckets.items():
        exact = n * len(members) / total
        quotas[key] = int(exact)
        remainders.append((exact - int(exact), key))

    allocated = sum(quotas.values())
    for _, key in sorted(remainders, key=lambda x: (-x[0], str(x[1]))):
        if allocated >= n:
            break
        if quotas[key] < len(buckets[key]):
            quotas[key] += 1
            allocated += 1

    sample = []
    for key in sorted(buckets, key=lambda k: (k[0], k[1])):
        take = min(quotas[key], len(buckets[key]))
        if take:
            sample.extend(rng.sample(buckets[key], take))

    sample.sort(key=lambda r: str(r["example_id"]))
    return sample
