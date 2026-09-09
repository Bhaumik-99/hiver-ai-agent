"""
Automated leakage checks. These RAISE rather than warn.

A benchmark that leaks does not produce a weaker result, it produces a
meaningless one, and the failure is silent — the previous version of this project
reported ROUGE-1 of 0.797 and a perfect 5.0/5.0 groundedness for its
nearest-neighbour baseline purely because the golden examples were still sitting
in the retrieval corpus. Every invariant that would have caught that is asserted
here and called from the harness before any number is reported.
"""


class LeakageError(AssertionError):
    """Raised when an evaluation-integrity invariant is violated."""


def assert_corpus_excludes_golden(corpus: list[dict], golden_set: list[dict]) -> dict:
    """
    The retrieval corpus must contain none of the messages being evaluated.

    Without this the top neighbour of a golden message is its own thread, so the
    generator writes its answer while looking at the reference answer.
    """
    golden_messages = {g["customer_message"].strip() for g in golden_set}
    overlap = [t for t in corpus if t.get("customer_message", "").strip() in golden_messages]
    if overlap:
        raise LeakageError(
            f"Retrieval corpus contains {len(overlap)} golden-set messages. "
            f"First offender: {overlap[0]['customer_message'][:120]!r}"
        )
    return {"check": "corpus_excludes_golden", "passed": True,
            "corpus_size": len(corpus), "golden_size": len(golden_set)}


def assert_retrieval_excludes_self(agent_results: list[dict],
                                   golden_set: list[dict]) -> dict:
    """
    No retrieved neighbour may be the evaluated example itself.

    This is the runtime counterpart of the corpus check: it catches leakage that
    slipped in through near-duplicate text or a corpus built somewhere else.
    """
    offenders = []
    for i, (res, gold) in enumerate(zip(agent_results, golden_set)):
        msg = gold["customer_message"].strip()
        ref = (gold.get("brand_reply") or "").strip()
        for conv in res.get("similar_conversations", []) or []:
            if conv.get("customer_message", "").strip() == msg:
                offenders.append((i, "retrieved its own customer message"))
                break
            if ref and conv.get("brand_reply", "").strip() == ref:
                offenders.append((i, "retrieved its own reference reply"))
                break
    if offenders:
        raise LeakageError(
            f"{len(offenders)} evaluated examples retrieved themselves. "
            f"First: index {offenders[0][0]} ({offenders[0][1]})."
        )
    return {"check": "retrieval_excludes_self", "passed": True,
            "n_checked": len(agent_results)}


def assert_baseline_out_of_fold(overrides: dict, golden_set: list[dict],
                                name: str) -> dict:
    """
    Every evaluated example must have an out-of-fold prediction.

    A baseline fitted on the whole golden set and then scored on part of it is
    reporting a training score. Cross-fitting is only meaningful if the harness
    actually uses the cross-fitted prediction for each row.
    """
    missing = [g["customer_message"] for g in golden_set
               if g["customer_message"] not in overrides]
    if missing:
        raise LeakageError(
            f"{name}: {len(missing)} evaluated examples have no out-of-fold "
            f"prediction, so they would be scored by a model fitted on them."
        )
    return {"check": f"{name}_out_of_fold", "passed": True,
            "n_predictions": len(overrides)}


def assert_judge_prompt_clean(prompt: str, reference_reply: str,
                              gold_intent: str = None) -> dict:
    """
    The judge prompt must not contain the reference reply or the gold label.

    Showing the judge the ground-truth answer turns it into a similarity metric
    and rewards verbatim copying.
    """
    ref = (reference_reply or "").strip()
    if ref and len(ref) > 25 and ref[:60] in prompt:
        raise LeakageError("Judge prompt contains the reference reply for this example.")
    if gold_intent and f'"{gold_intent}"' in prompt:
        raise LeakageError("Judge prompt contains the gold intent label.")
    return {"check": "judge_prompt_clean", "passed": True}


def assert_no_gold_labels_in_inference(agent_results: list[dict],
                                       golden_set: list[dict]) -> dict:
    """
    Nothing the agent produced may contain the reference reply verbatim.

    Catches the case where a gold label or reference answer reached the model
    through some path other than retrieval.
    """
    offenders = []
    for i, (res, gold) in enumerate(zip(agent_results, golden_set)):
        ref = (gold.get("brand_reply") or "").strip()
        reply = (res.get("reply") or "").strip()
        if ref and len(ref) > 40 and ref == reply:
            offenders.append(i)
    if offenders:
        raise LeakageError(
            f"{len(offenders)} generated replies are byte-identical to the "
            f"reference reply (indices {offenders[:5]}). Either the corpus still "
            f"contains the golden threads or the reference reached the prompt."
        )
    return {"check": "no_gold_labels_in_inference", "passed": True,
            "n_checked": len(agent_results)}


def assert_same_examples(runs: dict) -> dict:
    """All systems must be scored on exactly the same examples, in the same order."""
    keys = list(runs.keys())
    if len(keys) < 2:
        return {"check": "same_examples", "passed": True, "n_systems": len(keys)}
    baseline = [r["customer_message"] for r in runs[keys[0]]]
    for k in keys[1:]:
        other = [r["customer_message"] for r in runs[k]]
        if other != baseline:
            raise LeakageError(
                f"System {k!r} was evaluated on different examples than {keys[0]!r} "
                f"({len(other)} vs {len(baseline)}). The comparison is invalid."
            )
    return {"check": "same_examples", "passed": True,
            "n_systems": len(keys), "n_examples": len(baseline)}


def run_all_checks(corpus, golden_set, runs, overrides) -> list[dict]:
    """
    Run every leakage invariant. Raises LeakageError on the first violation.
    Returns the list of passed-check records for the run config.
    """
    records = [assert_corpus_excludes_golden(corpus, golden_set),
               assert_same_examples(runs)]
    for name, ov in (overrides or {}).items():
        records.append(assert_baseline_out_of_fold(ov, golden_set, name))
    for name, results in runs.items():
        records.append({**assert_retrieval_excludes_self(results, golden_set),
                        "system": name})
        records.append({**assert_no_gold_labels_in_inference(results, golden_set),
                        "system": name})
    return records
