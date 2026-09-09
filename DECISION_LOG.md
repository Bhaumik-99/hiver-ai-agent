# Decision Log

Non-obvious decisions, and why. Each one names the alternative that was rejected.
Where a decision was made *because* something turned out to be broken, that is
stated plainly rather than presented as foresight.

---

### 1. Brand: AppleSupport

**Alternatives:** AmazonHelp (larger volume), Uber_Support (higher emotional range),
SpotifyCares (narrower, easier domain).

AppleSupport was chosen for a disciplined and highly recognisable reply protocol —
acknowledge, request the OS/device version, move private matters to DM, sign with
agent initials (`^AL`). A consistent house style is what makes "grounded in how
this brand historically replied" a testable claim rather than a vague one. The
domain also spans genuinely different risk levels (a feature question versus a
swelling battery), which the escalation component needs in order to be interesting.

### 2. Threads are first-turn pairs, not full conversations

`scripts/process_brand.py` reconstructs one `(customer message, first brand reply)`
pair per conversation and hard-codes `num_turns: 2`. Multi-turn reconstruction was
rejected for this scope.

The consequence is honest but real: the escalation module accepts a
`conversation_history` argument that is always `None` in evaluation, so the
"customer has been going back and forth without resolution" signal in its prompt
can never fire on this data. Repeat-contact frustration is therefore **out of
scope and unmeasured**, not solved.

### 3. Taxonomy derived from the data, not from Banking77

Banking77's 77 intents describe retail banking products and map onto almost none
of Apple's device-support traffic. An LLM pass over a sample of real messages
produced 10 categories chosen to be *actionable* — each one implies a different
next step for a support queue, which is what a routing label is for.

`general_inquiry_features` is an explicit catch-all. It absorbs 43.5% of the
pre-labelled set, which is a weakness of the taxonomy and is reported as such
rather than hidden.

### 4. TF-IDF retrieval rather than embeddings

Rejected: sentence-transformer embeddings, hybrid BM25 + dense.

TF-IDF with bigrams and sublinear term frequency runs locally in under a second,
needs no embedding API and no extra daily quota, and is genuinely competitive on
short texts full of exact product nouns ("iOS 11", "AirPods", "iCloud"). Its
weakness — no paraphrase matching — is a named item in the one-week plan.

### 5. Golden-set labels are *pre-labels* until a human says otherwise

This is the most important decision in the project, and it reverses what the
repository originally claimed.

`scripts/label_golden_set.py` generates labels by regex. The original submission
wrote those to `golden_set_labelled.csv` and described them in the report as
"200 Hand-Audited" examples with a dual-annotator Cohen's κ of 0.87. No human
had reviewed anything, and no κ had been computed anywhere in the code.

The data is now staged so provenance lives in the filesystem and in the rows
themselves, not in prose: `golden_set_raw.csv` → `golden_set_prelabelled.csv`
(every row carries `annotation_status=pre_labelled_heuristic`) →
`golden_set_final.csv`, which only `scripts/finalize_golden_set.py` writes and
only from rows a human marked `human_reviewed`.

### 6. The harness refuses to run on pre-labels

`src/golden_set.load_evaluation_set` raises `GoldenSetProvenanceError` unless the
set is fully human-reviewed. `--allow-prelabelled` overrides it and stamps
`labels_are_provisional: true` into `run_config`, `comparison_*.json`, and the
generated results block in both README and REPORT.

Rejected: a warning line. A warning at the top of a long log is not read; a
refusal is. The cost — you cannot get a number without consciously opting into a
provisional one — is the entire point.

### 7. A status flip does not count as review

Added after a `golden_set_final.csv` appeared mid-audit with all 200 rows marked
`human_reviewed` while every label was byte-identical to the regex output and
every row still carried the generator's own provenance note.

`detect_unreviewed_relabelling` compares the claim against the pre-labels and
refuses a file that agrees with the machine on *every* row while retaining its
marker. A real reviewer agrees with a pre-label often; they do not agree with it
200 times out of 200 and leave the marker on each one. The offending file is in
`data/quarantine/` with an explanation, and `tests/test_golden_set.py` has the
regression test.

### 8. Golden examples are removed from the retrieval corpus

The golden set was sampled *from* the same 5,000 threads used for retrieval, so
the nearest neighbour of a golden message was its own thread. The
nearest-neighbour baseline was therefore returning the reference reply verbatim
and scoring ROUGE-1 0.797 with a groundedness of 5.00/5.00 at zero variance —
a measurement of the leak, not of the system. With the corpus filtered, the same
baseline scores ROUGE-1 ≈ 0.29.

All 200 golden messages are held out for both the `headline` and `full`
benchmarks, so the two share one corpus and stay comparable.

### 9. Leakage invariants raise instead of warning

`src/leakage_checks.py` asserts seven invariants and the run aborts with exit
code 4 rather than printing a leaked number. The bug in decision 8 was silent for
an entire submission cycle; a guard that only warns would have been equally
silent. `tests/test_leakage.py` builds a deliberately leaked scenario for each
invariant and asserts it raises — a guard that never fires is not a guard.

### 10. Baselines are scored on out-of-fold predictions

The TF-IDF classifier was originally fitted on the entire golden set and then
evaluated on a slice of it, so its reported accuracy was a training score.
It is now cross-fitted with `StratifiedKFold`, which keeps all 200 rows evaluable
while guaranteeing no row is predicted by a model that saw its label. Fold count
is capped by the smallest intent class (4 examples), so 4 folds.

Rejected: a held-out split, which would have shrunk an already small evaluation
set.

### 11. The trivial baseline had to be repaired before it was a baseline

It was fitted on `["dummy"] * len(intents)` paired with the *deduplicated*
taxonomy names, so every class tied at count 1 and `most_common()` returned
whichever intent sat first in the taxonomy file. It predicted
`software_update_os`, not the true majority `general_inquiry_features` (43.5%).

That understated the floor and inflated every "improvement over trivial" claim.
`TrivialAgent` now requires the observed training label sequence and raises
without it.

### 12. Macro-F1 is computed over the declared taxonomy

`sklearn` inferred the label set from `set(gold + predictions)`, so a system that
used more of the taxonomy was scored over a larger denominator with extra
support-0 classes contributing F1 = 0. The trivial baseline was averaged over 6
classes and the main agent over 8 — not a comparison. The taxonomy is now passed
explicitly, and off-taxonomy predictions are counted and reported instead of
being silently dropped by the metric.

### 13. The judge is blind to the reference reply

It was previously shown the brand's actual reply while scoring "groundedness",
which makes it a similarity metric: the nearest-neighbour baseline scored a
perfect 5.00 for reproducing the text it had been handed. The judge now sees only
a fixed set of three brand-voice exemplars, **identical for every system**, drawn
from the leakage-free corpus. Previously each system was judged against its own
retrieved neighbours, which handed one baseline its own output as the standard.

### 14. The judge runs on a different model from the agent

Generator and judge were both `gpt-oss-120b`. A model scoring its own family's
output rates it higher than a neutral grader does. They are now separate
(`JUDGE_GROQ_MODEL` vs `GROQ_MODEL`) and `run_config.judge_differs_from_agent`
records whether the separation actually held for a given run — with a single
local Ollama model it cannot, and the harness prints a warning saying so.

### 15. Judge calibration against a regex is not human agreement

The original calibration compared the judge to the `gold_reply_quality` column,
which `scripts/label_golden_set.py` computes from keyword presence. That measures
judge-versus-regex agreement. It was removed rather than relabelled, and replaced
with `build_judge_calibration_task.py` / `compute_judge_calibration.py`, which
export replies sampled across all three systems for a human to score and then
compute Spearman, exact and adjacent agreement, MAE and quadratic-weighted kappa.

Until a human fills that file the result is written as `status: PENDING` and the
requirement is marked **PARTIAL** in `SUBMISSION_AUDIT.md`. Quadratic weighting
is used because unweighted kappa treats 4-vs-5 as badly as 1-vs-5 on an ordinal
scale.

### 16. Escalation keywords are phrases, and mild profanity does not escalate

The original list escalated on bare `news`, `fire`, `burn`, `hurt`, `sue`,
`court`, `damn`, `crap`, `wtf`, `stolen`. `"news"` fires on *"any news on the
update?"*. Mild expletives are ordinary register in this dataset, and escalating
on them buries genuine cases in false positives.

Two rules were *added* because a real support desk would have them: an explicit
request for a human, and any billing dispute. `tests/test_escalation.py` asserts
both the positive cases and ten ordinary messages that must **not** escalate, plus
a guard that fails if any bare high-frequency word is reintroduced.

Severe ambiguity is deliberately **not** a deterministic rule — short messages
like *"see, it did it again"* are escalation-worthy but not keyword-detectable —
so it is delegated to the LLM layer, and a test documents that gap.

### 17. Headline benchmark is 40 stratified examples, seeded

The full 200-example run cannot meet the assignment's 15-minute reproduction
budget on a rate-limited free tier. Rather than claim it does, there are two
named benchmarks: `headline` (40 examples, the README command) and `full` (200,
extended). The subsample is jointly stratified on escalation flag and intent with
seed 42, so it holds the 34% escalation base rate — the previous `rows[:n]` slice
gave 10% and described a different distribution from the one it claimed.

### 18. Provider choice is an environment variable, and the reason is unglamorous

Every Groq model on the free tier carries its own tokens-per-day budget, and a
full benchmark plus judging exhausts one. `GROQ_MODEL` and `JUDGE_GROQ_MODEL`
exist so a run can move to a model that still has headroom without editing code,
and `run_config_*.json` records exactly what ran. A client-side pacer reads the
`x-ratelimit-remaining-tokens` header and sleeps precisely, instead of
discovering the limit through 429s; a tokens-per-*day* exhaustion now fails fast
with instructions, because no backoff will clear it.

### 19. Every documented number is generated, not typed

`scripts/render_results.py` renders the results tables from
`results/comparison_*.json` into marker blocks in README.md and REPORT.md, and
`--check` fails if they drift. The original report quoted an 82.5% intent
accuracy, a 0.864 escalation F1 and a 4.62/5 judge score that appeared in no
results file; the run that did exist had `samples_evaluated: 10` and an
escalation F1 of 0.0. Hand-typed numbers are how that happens.

### 20. Failures are recorded, not suppressed

An empty generation used to be skipped by the ROUGE loop, so a system that failed
to answer 17% of messages had its average computed over the 83% it did answer.
Empty predictions now score 0. Generation failures return `""` and are counted
rather than crashing a long run, `reply_validation` counts every observed
violation by type, and the failure analysis refuses to pad to five cases — it
prints how many genuine failures there actually were.
