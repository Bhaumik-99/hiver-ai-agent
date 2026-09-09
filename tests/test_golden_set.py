"""
Golden-set schema, provenance and deterministic-sampling tests.

The provenance tests are the important ones: they assert that the harness cannot
be tricked into reporting a benchmark against machine-generated labels.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.golden_set import (
    FINAL_PATH, PRELABELLED_PATH, SCHEMA, STATUS_HUMAN, STATUS_PENDING,
    STATUS_PRELABELLED, GoldenSetProvenanceError, annotation_summary,
    load_evaluation_set, read_rows, stratified_subsample, validate_schema,
    write_rows,
)


def _row(i, intent="general_inquiry_features", escalate=False,
         status=STATUS_HUMAN):
    return {
        "example_id": str(i),
        "customer_message": f"message {i}",
        "brand_reply": f"reply {i}",
        "difficulty": "standard",
        "gold_intent": intent,
        "gold_should_escalate": escalate,
        "gold_escalation_reason": "legal_threat" if escalate else "",
        "gold_reply_quality": 3,
        "labelling_notes": "",
        "annotation_status": status,
    }


# ── Schema ────────────────────────────────────────────────────────────

def test_prelabelled_file_exists_and_matches_schema():
    rows = read_rows(PRELABELLED_PATH)
    assert len(rows) == 200, f"expected 200 golden examples, found {len(rows)}"
    assert validate_schema(rows) == []


def test_schema_has_every_required_column():
    required = {
        "example_id", "customer_message", "brand_reply", "difficulty",
        "gold_intent", "gold_should_escalate", "gold_escalation_reason",
        "gold_reply_quality", "labelling_notes", "annotation_status",
    }
    assert required == set(SCHEMA)


def test_validate_schema_catches_problems():
    bad = [_row(1), _row(1)]                     # duplicate id
    assert any("duplicate" in p for p in validate_schema(bad))

    bad = [_row(1)]; bad[0]["gold_intent"] = ""
    assert any("gold_intent" in p for p in validate_schema(bad))

    bad = [_row(1)]; bad[0]["annotation_status"] = "made_up"
    assert any("annotation_status" in p for p in validate_schema(bad))

    bad = [_row(1)]; bad[0]["gold_reply_quality"] = 9
    assert any("gold_reply_quality" in p for p in validate_schema(bad))


def test_golden_set_size_is_in_assignment_range():
    rows = read_rows(PRELABELLED_PATH)
    assert 150 <= len(rows) <= 250


# ── Provenance enforcement ────────────────────────────────────────────

def test_prelabels_are_not_marked_human_reviewed():
    """The heuristic output must never claim to be human work."""
    rows = read_rows(PRELABELLED_PATH)
    human = [r for r in rows if r["annotation_status"] == STATUS_HUMAN]
    assert not human, (
        f"{len(human)} heuristic rows are marked {STATUS_HUMAN}; "
        "pre-labels must not be presented as human annotations")


def test_harness_refuses_prelabels_by_default():
    if os.path.exists(FINAL_PATH):
        pytest.skip("a human-reviewed final set exists; refusal path not applicable")
    with pytest.raises(GoldenSetProvenanceError):
        load_evaluation_set(allow_prelabelled=False)


def test_prelabelled_run_is_flagged_provisional():
    if os.path.exists(FINAL_PATH):
        pytest.skip("a human-reviewed final set exists")
    rows, prov = load_evaluation_set(allow_prelabelled=True)
    assert prov["labels_are_provisional"] is True
    assert "PROVISIONAL" in prov["warning"]
    assert prov["human_reviewed"] == 0


def test_annotation_summary_counts(tmp_path):
    rows = [_row(1), _row(2, status=STATUS_PRELABELLED), _row(3, status=STATUS_PENDING)]
    s = annotation_summary(rows)
    assert s == {"total": 3, "human_reviewed": 1, "pre_labelled_heuristic": 1,
                 "pending_review": 1, "fully_human_reviewed": False}


def test_fully_human_set_is_not_provisional(tmp_path):
    rows = [_row(i) for i in range(1, 21)]
    p = tmp_path / "final.csv"
    write_rows(rows, str(p))
    loaded, prov = load_evaluation_set(allow_prelabelled=False, path=str(p))
    assert prov["labels_are_provisional"] is False
    assert prov["human_reviewed"] == 20


# ── Deterministic sampling ────────────────────────────────────────────

def test_subsample_is_deterministic():
    rows = read_rows(PRELABELLED_PATH)
    a = [r["example_id"] for r in stratified_subsample(rows, 40, seed=42)]
    b = [r["example_id"] for r in stratified_subsample(rows, 40, seed=42)]
    assert a == b, "same seed must give the same benchmark set"


def test_subsample_seed_changes_selection():
    rows = read_rows(PRELABELLED_PATH)
    a = [r["example_id"] for r in stratified_subsample(rows, 40, seed=42)]
    c = [r["example_id"] for r in stratified_subsample(rows, 40, seed=7)]
    assert a != c


def test_subsample_size_is_exact():
    rows = read_rows(PRELABELLED_PATH)
    for n in (20, 40, 60):
        assert len(stratified_subsample(rows, n, seed=42)) == n


def test_subsample_preserves_escalation_base_rate():
    """
    The headline benchmark must describe the same distribution as the full set.
    Slicing rows[:n] gave a 10% escalation rate against the full set's 34%.
    """
    rows = read_rows(PRELABELLED_PATH)
    full_rate = sum(r["gold_should_escalate"] for r in rows) / len(rows)
    sample = stratified_subsample(rows, 40, seed=42)
    sample_rate = sum(r["gold_should_escalate"] for r in sample) / len(sample)
    assert abs(sample_rate - full_rate) < 0.06, (
        f"escalation base rate drifted: {sample_rate:.1%} vs {full_rate:.1%}")


def test_subsample_covers_multiple_intents():
    rows = read_rows(PRELABELLED_PATH)
    sample = stratified_subsample(rows, 40, seed=42)
    intents = {r["gold_intent"] for r in sample}
    assert len(intents) >= 5, f"benchmark covers only {len(intents)} intents"


def test_subsample_returns_all_when_n_exceeds_size():
    rows = read_rows(PRELABELLED_PATH)
    assert len(stratified_subsample(rows, 10_000, seed=42)) == len(rows)


# ── Anti-laundering ───────────────────────────────────────────────────

def test_status_flip_alone_does_not_make_labels_human(tmp_path):
    """
    Regression test for a real incident during the audit: a golden_set_final.csv
    appeared with all 200 rows marked human_reviewed while every label was
    byte-identical to the heuristic output. Flipping the status column must not
    be enough to pass the provenance gate.
    """
    from src.golden_set import HEURISTIC_MARKER, detect_unreviewed_relabelling

    pre = read_rows(PRELABELLED_PATH)
    laundered = []
    for r in pre:
        row = dict(r)
        row["annotation_status"] = STATUS_HUMAN      # the only change
        laundered.append(row)

    check = detect_unreviewed_relabelling(laundered)
    assert check["suspicious"] is True
    assert check["n_identical_to_prelabels"] == check["n_claiming_human_review"]

    p = tmp_path / "laundered.csv"
    write_rows(laundered, str(p))
    with pytest.raises(GoldenSetProvenanceError, match="status column rewritten"):
        load_evaluation_set(allow_prelabelled=False, path=str(p))


def test_genuine_review_that_changes_labels_is_accepted(tmp_path):
    """A real review corrects some labels and drops the generator's marker."""
    from src.golden_set import detect_unreviewed_relabelling

    pre = read_rows(PRELABELLED_PATH)[:30]
    reviewed = []
    for i, r in enumerate(pre):
        row = dict(r)
        row["annotation_status"] = STATUS_HUMAN
        row["labelling_notes"] = "reviewed by annotator A"
        if i % 3 == 0:                                # genuine corrections
            row["gold_intent"] = "battery_and_charging"
        reviewed.append(row)

    assert detect_unreviewed_relabelling(reviewed)["suspicious"] is False
    p = tmp_path / "reviewed.csv"
    write_rows(reviewed, str(p))
    rows, prov = load_evaluation_set(allow_prelabelled=False, path=str(p))
    assert prov["labels_are_provisional"] is False


def test_no_final_golden_set_is_committed_unreviewed():
    """
    The repository must not ship a golden_set_final.csv that is merely the
    pre-labels renamed. If this fails, someone laundered the labels again.
    """
    from src.golden_set import detect_unreviewed_relabelling
    if not os.path.exists(FINAL_PATH):
        pytest.skip("no final set present, which is the honest state pre-annotation")
    check = detect_unreviewed_relabelling(read_rows(FINAL_PATH))
    assert not check["suspicious"], (
        "data/golden_set_final.csv claims human review but is identical to the "
        "heuristic pre-labels")
