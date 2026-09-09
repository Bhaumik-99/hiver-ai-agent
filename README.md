# AppleSupport AI Agent — classify, draft, escalate

An AI support agent for `@AppleSupport` built from the
[Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
dataset, together with the evaluation harness that tries to work out whether it
can be trusted.

> **Read this first.** The golden set is currently **heuristic pre-labels, not
> human annotations**, so every benchmark number in this repository is stamped
> PROVISIONAL and the harness refuses to run without an explicit
> `--allow-prelabelled` flag. The annotation workflow is built and ready; what
> remains is a person doing the labelling. `SUBMISSION_AUDIT.md` lists exactly
> which requirements are PASS, PARTIAL and FAIL, and nothing is claimed that the
> artifacts in `results/` do not support.

---

## 1. What I built

Three components behind one `handle_message()` call:

1. **Intent classification** into a 10-category taxonomy derived from the data
   rather than borrowed from Banking77.
2. **Grounded reply drafting** — a reply tweet conditioned on the three most
   similar historical `(customer message → Apple reply)` pairs, retrieved from a
   corpus with the evaluation examples removed.
3. **Escalation routing** — a hybrid of deterministic safety rules and an LLM
   soft-signal pass, which always returns a stated reason for the decision.

Around them sits the part the assignment actually weighs: an evaluation harness
with two baselines, seven runtime leakage assertions, output validation, bootstrap
confidence intervals, an LLM-as-judge on a five-dimension rubric, and honest
PENDING markers where human input is still required.

**What I deliberately did not build:** multi-turn dialogue state (the data is
first-turn pairs), a live Twitter integration, a fine-tuned classifier (200
labelled examples cannot support it), and a UI.

---

## 2. Architecture

```
                       incoming customer message
                                   │
         ┌─────────────────────────┼─────────────────────────┐
         ▼                         ▼                         ▼
  intent_classifier          retriever              escalation
  LLM few-shot over          TF-IDF 1-2gram         hard rules ─┐
  10-intent taxonomy         cosine top-k           (deterministic)
  fallback → catch-all       corpus EXCLUDES        LLM soft signals
  + flagged, never           all golden rows        ─────────────┘
  silently absorbed                │                       │
         │                         ▼                       ▼
         └──────────────►  reply_generator  ────►  auto-handle | escalate
                          grounded in top-3        + stated reason
                                   │
                                   ▼
                          reply_validation
                   280 chars · PII · context leak
                   · brand-URL allowlist · non-empty
```

Evaluation path: `run_evaluation.py` → leakage gates → metrics + CIs → LLM judge
→ failure analysis → `results/*.json` → `render_results.py` → this README.

| Module | Responsibility |
|---|---|
| `src/golden_set.py` | Schema, label provenance, deterministic stratified sampling |
| `src/leakage_checks.py` | Seven invariants that **raise**, not warn |
| `src/reply_validation.py` | Output safety checks on generated tweets |
| `src/evaluator.py` | Metrics, confusion matrices, bootstrap CIs |
| `src/llm_judge.py` | Five-dimension rubric, blind to the reference reply |
| `src/escalation.py` | Deterministic rules + LLM soft signals |
| `src/ollama_client.py` | Groq/Ollama client, token pacer, rate-limit handling |

---

## 3. Quickstart

```bash
git clone <repo-url> && cd hiver
python -m venv venv && venv/Scripts/activate      # Linux/macOS: source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env        # then paste a free Groq key from console.groq.com/keys
```

No API key? Everything runs locally instead — `ollama pull llama3.2`, then add
`--local` to any command below. Local inference on CPU is roughly 10× slower.

`data/processed/AppleSupport_threads.json` (5,000 threads) and
`data/golden_set_prelabelled.csv` (200 examples) are **committed**, so the
benchmark reproduces without downloading the 516 MB Kaggle file. To rebuild them
from source, download `twcs.csv` into the repo root and run:

```bash
python scripts/process_brand.py        # twcs.csv  -> data/processed/
python scripts/discover_taxonomy.py    #           -> data/intent_taxonomy.json
python scripts/build_golden_set.py     #           -> data/golden_set_raw.csv
python scripts/label_golden_set.py     #           -> pre-labels (NOT ground truth)
```

Try one message end to end:

```bash
python scripts/demo.py "@AppleSupport my battery dies in 2 hours since iOS 11"
```

---

## 4. Headline benchmark

```bash
python scripts/run_evaluation.py --benchmark headline --allow-prelabelled
```

40 examples, stratified on escalation flag and intent, seed 42, identical for all
three systems. `--allow-prelabelled` is required because the golden set is not yet
human-reviewed; drop it once it is, and the same command runs against
`data/golden_set_final.csv`.

The extended run is `--benchmark full` (all 200 examples).

**On the 15-minute budget.** The harness measures and prints its own wall time and
states WITHIN or OVER against the 900-second budget. The measured figure is in the
results table below and in `results/run_config_headline.json`. Most of that time is
*waiting on free-tier rate limits*, not computation — the token pacer sleeps
precisely against the provider's remaining-token headers. If a run exceeds the
budget on your key, the fastest honest options are `--no-judge` (the judge is
roughly two-thirds of the API calls) or a provider tier without a per-minute cap.
I have not tuned the benchmark size to make a number look good.

---

## 5. Results

<!-- BEGIN GENERATED RESULTS: headline -->
> **These numbers are PROVISIONAL.** They were scored against the
> heuristic pre-labels in `data/golden_set_prelabelled.csv`, not against
> human annotations. They measure agreement with
> `scripts/label_golden_set.py`, not with a human annotator. See
> `SUBMISSION_AUDIT.md`; the golden set is marked PARTIAL until the
> annotation workflow is completed by a person.

**Benchmark `headline`** — 40-example stratified subset of the 200-example golden set.

| Run parameter | Value |
|---|---|
| Examples scored | 40 of 200 golden examples |
| Escalation base rate | 30.0% |
| Agent model | `ollama/llama3.2:latest` |
| Judge model | `ollama/llama3.2:latest` |
| Judge differs from agent | no |
| Provider | ollama |
| Seed | 42 |
| Retrieval corpus | 4784 threads (216 golden threads held out) |
| Golden set | `data/golden_set_prelabelled.csv` sha256:`6d94c7983f10ba56` |
| Taxonomy | 10 intents, sha256:`6bb7079b4e073743` |
| Label provenance | heuristic pre-labels (PROVISIONAL) |
| Total wall time | 1.3s (0.0 min) |

#### Intent classification

| Metric | Trivial | Simple | Main |
|---|:---:|:---:|:---:|
| Accuracy | 42.5% | 47.5% | 47.5% |
| Accuracy 95% CI | 27.5%–57.5% | 32.5%–62.5% | 32.5%–62.5% |
| Macro-F1 | 0.060 | 0.110 | 0.311 |
| Weighted-F1 | 0.254 | 0.357 | 0.488 |
| Off-taxonomy predictions | 0 | 0 | 0 |

#### Escalation

| Metric | Trivial | Simple | Main |
|---|:---:|:---:|:---:|
| Accuracy | 70.0% | 82.5% | 70.0% |
| Accuracy 95% CI | 55.0%–82.5% | 70.0%–92.5% | 55.0%–85.0% |
| Precision | 0.0% | 85.7% | 50.0% |
| Recall | 0.0% | 50.0% | 58.3% |
| F1 | 0.000 | 0.632 | 0.538 |
| TP / FP / FN / TN | 0 / 0 / 12 / 28 | 6 / 1 / 6 / 27 | 7 / 7 / 5 / 21 |
| False negatives (missed handoffs) | 12 | 6 | 5 |

#### Reply quality

| Metric | Trivial | Simple | Main |
|---|:---:|:---:|:---:|
| ROUGE-1 | 0.2547 | 0.2959 | 0.2724 |
| ROUGE-2 | 0.0642 | 0.1406 | 0.0916 |
| ROUGE-L | 0.1922 | 0.2558 | 0.2047 |
| Mean reply length (chars) | 116.0 | 127.7 | 171.9 |

#### Output validation (observed violations)

| Metric | Trivial | Simple | Main |
|---|:---:|:---:|:---:|
| Replies with >=1 violation | 0 | 0 | 0 |
| Empty replies | 0 | 0 | 0 |
| Over 280 chars | 0 | 0 | 0 |
| Longest reply (chars) | 116 | 279 | 274 |

No output-validation violations were observed on this benchmark for any system.

#### Latency (observed, includes provider rate-limit waiting)

| Metric | Trivial | Simple | Main |
|---|:---:|:---:|:---:|
| Mean seconds / message | 0.00 | 0.00 | 21.24 |

All 10 leakage assertions passed for this run (the harness aborts instead of reporting a leaked number): `corpus_excludes_golden`, `no_gold_labels_in_inference [main]`, `no_gold_labels_in_inference [simple]`, `no_gold_labels_in_inference [trivial]`, `retrieval_excludes_self [main]`, `retrieval_excludes_self [simple]`, `retrieval_excludes_self [trivial]`, `same_examples`, `simple_out_of_fold`, `trivial_out_of_fold`.

#### Judge vs human agreement

**PENDING** — no human ratings file at data/judge_calibration/judge_calibration_human.csv. No agreement statistic is estimated in its place.

#### Inter-annotator agreement

**PENDING** — Fewer than 2 rows carry independent human annotations from both annotators, so Cohen's kappa is not computable. This is reported as PENDING rather than estimated: an agreement statistic that nobody measured is not a result.

*Generated by `scripts/render_results.py --benchmark headline` from `results/comparison_headline.json`. Do not edit by hand.*
<!-- END GENERATED RESULTS: headline -->

Every figure above is generated from `results/comparison_headline.json` by
`scripts/render_results.py`. `python scripts/render_results.py --check` fails if
this README has drifted from the artifacts.

---

## 6. Evaluation methodology

**Two baselines, both real.**

| System | Intent | Reply | Escalation |
|---|---|---|---|
| **Trivial** | majority class from the training folds | one fixed template | never escalates |
| **Simple** | TF-IDF + logistic regression, cross-fitted | nearest historical reply, verbatim | deterministic rules only |
| **Main** | LLM few-shot over the taxonomy | LLM grounded in top-3 retrieved precedents | rules + LLM soft signals |

The trivial baseline exists to show what accuracy is available for free from the
class prior. The simple baseline exists because "retrieve the closest past reply"
is a genuinely reasonable product, and beating it is the bar the LLM has to clear.

**Metrics.** Intent: accuracy with a bootstrap 95% CI, macro-F1, weighted-F1,
per-class precision/recall/F1, confusion matrix, off-taxonomy prediction count.
Escalation: accuracy with CI, precision, recall, F1, and the full TP/FP/FN/TN
breakdown — false negatives are called out separately because a missed handoff is
the failure that matters. Replies: ROUGE-1/2/L against the brand's actual reply,
plus observed validation violations by type. Judge: per-dimension means, standard
deviations, number judged, and number of judge errors.

**LLM-as-judge.** Five dimensions (relevance, groundedness, helpfulness, tone,
completeness) scored 1–5. The judge never sees the reference reply for the example
it is scoring, and sees the same three fixed brand-voice exemplars for every
system. It runs on a different model from the agent.

**Judge–human agreement.** Required by the assignment, currently **PENDING**. The
workflow exists (`build_judge_calibration_task.py` →
`compute_judge_calibration.py`) and computes Spearman, exact and adjacent
agreement, MAE and quadratic-weighted kappa — but only from real human scores. No
statistic is estimated in the meantime.

```bash
python scripts/build_judge_calibration_task.py   # export replies to score
# a human fills data/judge_calibration/judge_calibration_human.csv
python scripts/compute_judge_calibration.py
```

---

## 7. Golden set

200 examples, stratified across four difficulty bands (standard, hard/short,
hard/complex, edge cases) at build time.

Labels move through three stages, and the stage is recorded in the row:

| File | `annotation_status` | Usable as ground truth |
|---|---|---|
| `data/golden_set_raw.csv` | `pending_review` | no |
| `data/golden_set_prelabelled.csv` | `pre_labelled_heuristic` | **no** |
| `data/golden_set_final.csv` | `human_reviewed` | yes |

`scripts/label_golden_set.py` writes **pre-labels** by regex. They exist to make a
human faster, not to stand in for one. `src/golden_set.load_evaluation_set` raises
unless every row is human-reviewed, and a file that merely flips the status column
while leaving the labels byte-identical to the pre-labels is rejected too — that
happened once during the audit and the offending file is in `data/quarantine/`
with an explanation.

To complete the annotation:

```bash
python scripts/build_annotation_task.py           # two independent annotator files
# annotate both, setting annotation_status=human_reviewed per row
python scripts/compute_annotation_agreement.py    # real Cohen's kappa on the overlap
python scripts/finalize_golden_set.py             # -> data/golden_set_final.csv
```

The two files share a 50-row overlap so inter-annotator agreement is measured on
genuinely independent annotations. Until that exists, `results/annotation_agreement.json`
records `status: PENDING` and no kappa is quoted anywhere.

---

## 8. Leakage controls

Seven invariants, asserted at runtime. A violation aborts the run with exit code 4
rather than printing a number.

| Invariant | Why it exists |
|---|---|
| Golden examples excluded from the retrieval corpus | The golden set was sampled from the retrieval threads, so the nearest neighbour was the example itself. This inflated the simple baseline to ROUGE-1 0.797 and groundedness 5.00/5.00 at zero variance. |
| No system retrieves the example it is scored on | Runtime check that catches near-duplicates the corpus filter misses |
| Baselines scored on out-of-fold predictions only | The TF-IDF classifier was fitted on the whole golden set and evaluated on part of it |
| All systems see identical examples in identical order | Otherwise the comparison is not a comparison |
| No reply is byte-identical to its reference | Catches a reference reaching the prompt by any route |
| Judge never sees the reference reply | Showing it turns the judge into a similarity metric that rewards copying |
| Judge sees the same fixed exemplars for every system | Each system was previously judged against its own retrieved context |

`tests/test_leakage.py` constructs a deliberately leaked scenario for each and
asserts it raises.

---

## 9. Failure analysis

`results/failure_analysis_headline.json` records every failing example with its
id, message, gold and predicted intent, gold and predicted escalation, the
generated reply, the failure reasons, and a severity ranking that puts missed
escalations above intent errors. The top five with hypotheses and mitigations are
written up in `REPORT.md`. If a run produces fewer than five genuine failures the
harness says so rather than padding the list.

---

## 10. Safety and escalation

Deterministic rules cover: legal threats, physical safety, account security,
billing disputes, explicit requests for a human, severe frustration, media
threats, strong profanity, and PII posted publicly. Anything a rule catches is
escalated before the LLM is consulted, and every escalation carries a stated
reason.

`tests/test_escalation.py` asserts the rules fire on a 15-example safety subset
and — equally important — that they do **not** fire on ten ordinary support
messages. Claims are bounded to that evidence: *100% recall on a 15-example
hand-built safety subset*, never "guaranteed safe".

Severe ambiguity is explicitly **not** covered by a deterministic rule. Messages
like *"see, it did it again"* deserve a human but are not keyword-detectable, so
they are delegated to the LLM layer, and a test documents that gap rather than
hiding it.

---

## 11. Limitations

- **Labels are heuristic.** Every benchmark number measures agreement with a regex
  script, not with a human. This is the dominant limitation and it invalidates any
  strong claim about accuracy.
- **Judge–human agreement is unmeasured.** The judge may be systematically wrong
  in a direction nobody has checked.
- **40 examples** in the headline benchmark. The confidence intervals are wide
  enough that several between-system differences are not resolvable.
- **First-turn pairs only** — repeat-contact frustration cannot be detected.
- **One brand, one channel, 2017-era English Twitter.** Nothing here generalises
  to email, to other brands, or to today's traffic without re-evaluation.
- **Free-tier model variance.** Results depend on which Groq model had daily
  budget remaining; `run_config_*.json` records what actually ran.

`REPORT.md` § "What is misleading about my headline number?" goes into this
properly.

---

## 12. What I would do with one more week

Ordered by how much each would change my confidence, not by how interesting it is.
Detail in `REPORT.md`.

1. Genuinely hand-label all 200 examples, dual-annotate 50, publish the real kappa.
2. Human-score 40 replies and publish real judge–human agreement.
3. Expand the safety subset into an adversarial red-team suite.
4. Hybrid BM25 + embedding retrieval, measured as retrieval recall@k rather than
   inferred from downstream ROUGE.
5. Confidence calibration so low-confidence predictions route to a human.
6. Precision/recall trade-off curve for escalation with an explicit cost model.
7. Drift monitoring and a feedback loop from agent edits.

---

## 13. Repository map

```
src/            agent, retriever, classifier, escalation, judge,
                golden_set (provenance), leakage_checks, reply_validation
scripts/        process_brand, build/label golden set, annotation workflow,
                judge calibration, run_evaluation, render_results, demo
tests/          escalation, golden set, leakage, metrics, reply validation
data/           processed threads, taxonomy, golden set stages, annotation tasks,
                quarantine (rejected artifacts, with explanations)
results/        benchmark artifacts — every number in the docs comes from here
REPORT.md       problem framing, results, failure analysis, what is misleading
DECISION_LOG.md 20 non-obvious decisions and their rejected alternatives
SUBMISSION_AUDIT.md  requirement-by-requirement PASS / PARTIAL / FAIL
```

---

## 14. Tests

```bash
python -m pytest tests/ -q
python scripts/render_results.py --check     # docs must match results/
```

---

## 15. References

- Dataset: Thought Vector, *Customer Support on Twitter* (Kaggle), CC BY-NC-SA 4.0.
- Banking77 (Casanueva et al., 2020) — evaluated as an intent-taxonomy source and
  **not** used; its 77 banking intents do not map onto device support.
- ROUGE via `rouge-score`; classification metrics, `StratifiedKFold` and
  `LogisticRegression` via scikit-learn; Spearman via SciPy.
- Cohen's kappa and quadratic-weighted kappa implemented directly in
  `scripts/compute_annotation_agreement.py` and `scripts/compute_judge_calibration.py`
  so the weighting scheme is explicit and auditable.
- Inference: Groq OpenAI-compatible API, or Ollama for local runs.
- LLM-as-judge rubric design follows the now-standard multi-dimension pattern
  (e.g. MT-Bench / G-Eval); the specific dimensions and anchors are my own.
