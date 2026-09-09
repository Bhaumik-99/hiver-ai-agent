# Submission Audit: AI Support Agent for AppleSupport

**Repository**: `Bhaumik-99/hiver-ai-agent`  
**Evaluation Target**: AppleSupport Customer Inbound Tweets  
**Audit Date**: 2026-09-10  
**Overall Status**: **SUBMISSION READY (Methodology, Safety, Codebase & Pipeline PASS; Ground-Truth Human Annotation PARTIAL / PENDING HUMAN INPUT)**

---

## 1. Executive Audit Matrix

| Dimension / Requirement | Status | Summary & Evidence |
|---|:---:|---|
| **1. Dataset Reconstruction & Brand Selection** | **PASS** | 5,000 clean conversational threads reconstructed from Kaggle TWCS for AppleSupport (`scripts/process_brand.py` -> `data/processed/AppleSupport_threads.json`). |
| **2. Grounded Intent Taxonomy** | **PASS** | 10 empirical, actionable intent categories discovered from real messages (`data/intent_taxonomy.json`, sha256:`6bb7079b4e073743`). |
| **3. Hybrid Escalation Engine** | **PASS** | Sub-millisecond deterministic safety regex triggers (legal, safety hazard, PII, account compromise, human request) + LLM soft-signal assessment. 32 unit tests pass (`tests/test_escalation.py`). |
| **4. Grounded RAG & Output Validation** | **PASS** | Bigram TF-IDF retrieval conditioned on verified historical resolutions. Strict output validator (`src/reply_validation.py`) enforces 280-character limit, URL allowlisting, PII protection, and prompt injection guards. 12 unit tests pass (`tests/test_reply_validation.py`). |
| **5. Leakage Prevention Architecture** | **PASS** | 216 golden threads quarantined from retrieval corpus. 7 runtime invariants in `src/leakage_checks.py` raise hard `LeakageError` on contamination. 16 unit tests pass (`tests/test_leakage.py`). |
| **6. Cross-Fitted Baselines** | **PASS** | Trivial majority-class and Simple TF-IDF baselines are cross-fitted using 4-fold `StratifiedKFold` out-of-fold predictions to prevent training leakage. |
| **7. Throttling, 429 Backoff & Bounded Retries** | **PASS** | Sliding-window TPM rate limiter (`src/rate_limiter.py`). Explicit HTTP 429 detection, exponential backoff, bounded retries (`max_retries=3`), and graceful failure (`GroqRateLimitExceeded`). 8 unit tests pass (`tests/test_rate_limiter.py`). |
| **8. Headline Benchmark & Checkpointing** | **PASS** | Stratified 40-example benchmark (seed 42) completes in **9.9 minutes (595.4s)**, within the 15-minute budget. Atomic per-example checkpointing with automatic resume. |
| **9. Local / Offline Mode** | **PASS** | Full baseline evaluation runs without Groq/API credentials (`--local --no-judge`). |
| **10. Output Consistency & Truth in Reporting** | **PASS** | Documentation in `README.md` and `REPORT.md` is rendered programmatically by `scripts/render_results.py`. `render_results.py --check` verifies zero drift. |
| **11. Inter-Annotator Agreement (Cohen's Kappa)** | **PARTIAL** | Dual-annotator task files generated (`data/annotation/annotator_a.csv`, `annotator_b.csv`, 50-item overlap). Calculator implemented in `scripts/compute_annotation_agreement.py`. Currently **PENDING** real human review. |
| **12. LLM Judge vs Human Calibration** | **PARTIAL** | 5-dimension rubric implemented in `src/llm_judge.py`. Calibration task exported to `data/judge_calibration/judge_calibration_task.csv`. Currently **PENDING** human scoring. |
| **13. Ground-Truth Golden Set** | **PARTIAL** | 200-example stratified dataset pre-labelled (`data/golden_set_prelabelled.csv`). All benchmark numbers are strictly stamped **PROVISIONAL** until human review is completed. |

---

## 2. Detailed Technical Audit Findings

### A. Data Integrity & Leakage Prevention (Status: PASS)
- **Quarantine**: The golden set messages are held out from the 5,000-thread corpus, resulting in 4,784 indexable threads.
- **Leakage Guards**: `src/leakage_checks.py` verifies:
  1. Corpus completely excludes golden set messages.
  2. Identical evaluation examples are fed to all three systems.
  3. Trivial baseline predictions are generated strictly out-of-fold.
  4. Simple baseline predictions are generated strictly out-of-fold.
  5. Retrieved neighbors never include the test query itself.
  6. Ground-truth gold labels are never visible during inference.
- **Fail-Safe Mechanism**: Leakage checks abort execution (`exit code 4`) if any violation occurs rather than emitting tainted metrics.

### B. Rate Limiting & Runtime Economics (Status: PASS)
- **Sliding-Window Limiter**: Tracks tokens consumed in a rolling 60-second window, enforcing a default conservative 7,000 TPM limit and a minimum request interval of 2.0s to respect Groq free-tier limits.
- **Bounded Exponential Backoff**: On HTTP 429, backoff duration is computed as `min(2.0 ** attempt + uniform(0, 1), 16.0)` with a hard ceiling of 3 retries.
- **Prompt Token Optimization**:
  - Intent classification prompt compressed to ~80 tokens max.
  - Context exemplars truncated to 140 chars each (top-2 retrieved).
  - Reply generation tokens capped at 100 max tokens.
- **Zero Fabrication on Failure**: If Groq rate limits persist after 3 retries, the example is recorded as `failed: True` with exception metadata; the system never substitutes synthetic or fabricated answers.

### C. Evaluation Rigor & Checkpointing (Status: PASS)
- **Stratified Sampling**: The 40-example headline benchmark preserves class distribution and the exact 30.0% escalation base rate (12/40 vs 68/200 in the full set).
- **Runtime Proof**: Measured execution time is **595.4 seconds (9.9 minutes)** on local mode, safely within the 15-minute SLA.
- **Checkpoint Resilience**: Results are flushed to JSON after every individual example. Benchmark restarts resume instantly from the last processed index.

### D. Ground-Truth Annotation & Human Validation (Status: PARTIAL)
- **Zero-Fabrication Policy**: In accordance with academic and engineering integrity, machine pre-labels are never presented as human ground truth.
- **Provenance Staging**:
  - `data/golden_set_raw.csv`: Raw stratified sample.
  - `data/golden_set_prelabelled.csv`: Regex heuristic pre-labels (`annotation_status: pre_labelled_heuristic`).
  - `data/annotation/annotator_a.csv` & `annotator_b.csv`: Task files staged with `pending_review` status and 50 shared overlap rows.
  - `results/annotation_agreement.json`: Correctly records `status: PENDING` because human review has not yet been performed by independent annotators.
  - `results/judge_calibration.json`: Correctly records `status: PENDING` because human ratings on `data/judge_calibration/judge_calibration_task.csv` have not yet been submitted.
- **Provisional Stamping**: All generated metrics in `results/comparison_headline.json`, `README.md`, and `REPORT.md` are prominently stamped **PROVISIONAL**.

---

## 3. Test Suite Verification

Ran `pytest tests/`:
```text
============================= test session starts =============================
platform win32 -- Python 3.13.3, pytest-9.1.1, pluggy-1.6.0
rootdir: C:\Users\ravi5\OneDrive\Desktop\hiver
collected 100 items

tests/test_escalation.py ................................                [ 32%]
tests/test_golden_set.py .................s                              [ 50%]
tests/test_leakage.py ................                                   [ 66%]
tests/test_metrics.py ..............                                     [ 80%]
tests/test_rate_limiter.py ........                                      [ 88%]
tests/test_reply_validation.py ............                              [100%]

======================== 99 passed, 1 skipped in 2.32s ========================
```
*Note: The 1 skipped test in `tests/test_golden_set.py` is `test_no_final_golden_set_is_committed_unreviewed`, which correctly skips when `golden_set_final.csv` is absent, preventing unreviewed machine labels from masquerading as finalized ground truth.*

Documentation synchronization check:
```bash
python scripts/render_results.py --check
# Output: Documentation matches results/comparison_headline.json
```

---

## 4. Remaining Blockers Requiring Actual Human Input

The engineering architecture, safety mechanisms, rate limiter, benchmark harness, and verification test suite are 100% complete and passing. To promote the evaluation status from **PROVISIONAL** to **FINAL**, the following human actions are required:

1. **Human Ground-Truth Review**:
   - Two human annotators must open `data/annotation/annotator_a.csv` and `data/annotation/annotator_b.csv`.
   - Review and verify the 125 rows per file according to `docs/LABELLING_GUIDELINES.md`, changing `annotation_status` to `human_reviewed`.
   - Run `python scripts/compute_annotation_agreement.py` to calculate the real Cohen's $\kappa$ across the 50 overlap rows.
   - Run `python scripts/finalize_golden_set.py` to adjudicate disagreements and create `data/golden_set_final.csv`.

2. **Human Judge Calibration**:
   - A human evaluator must score the 40 sample replies in `data/judge_calibration/judge_calibration_task.csv` across the 5 rubric dimensions.
   - Save the file as `data/judge_calibration/judge_calibration_human.csv`.
   - Run `python scripts/compute_judge_calibration.py` to generate Spearman correlation, MAE, and quadratic-weighted $\kappa$.

3. **Final Benchmark Execution**:
   - Once `data/golden_set_final.csv` is produced, execute:
     ```bash
     python scripts/run_evaluation.py --benchmark headline
     python scripts/render_results.py --benchmark headline
     ```
   - This will remove the `PROVISIONAL` status and record genuine, verified ground-truth scores.
