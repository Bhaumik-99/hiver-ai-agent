"""
Evaluation harness: runs all three systems on one fixed benchmark and reports
metrics, LLM-judge scores, reply validation, and failure analysis.

    python scripts/run_evaluation.py --benchmark headline   # ~40 examples, target <15 min
    python scripts/run_evaluation.py --benchmark full       # all 200, extended
    python scripts/run_evaluation.py --benchmark headline --no-judge

Evaluation-integrity invariants, all ASSERTED at runtime by src/leakage_checks
(the run aborts rather than reporting a leaked number):
  1. Golden examples are removed from the retrieval corpus.
  2. No system retrieves the example it is being scored on.
  3. Baseline classifiers are scored on out-of-fold predictions only.
  4. All systems see exactly the same examples in the same order.
  5. No generated reply is byte-identical to its reference reply.
  6. The judge never sees the reference reply; it sees a fixed, system-independent
     set of brand-voice exemplars, identical for every system.
  7. Gold labels are never used during inference.

Label provenance is enforced by src/golden_set: if the golden set is not fully
human-reviewed the harness refuses to run unless --allow-prelabelled is passed,
and every artifact it then writes is stamped PROVISIONAL.
"""
import argparse
import json
import os
import random
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from src.data_processor import load_processed_data
from src.intent_classifier import (
    TfidfClassifier, TrivialClassifier,
    load_taxonomy, discover_intents_from_data, save_taxonomy,
)
from src.agent import SupportAgent, TrivialAgent, SimpleAgent
from src.evaluator import full_evaluation, format_results_table, save_results
from src.llm_judge import judge_batch, aggregate_judgments, JUDGE_GROQ_MODEL
from src.ollama_client import check_ollama, check_groq, DEFAULT_GROQ_MODEL
from src.golden_set import (
    GoldenSetProvenanceError, file_hash, load_evaluation_set, stratified_subsample,
)
from src.leakage_checks import LeakageError, run_all_checks
from src.reply_validation import validate_batch, validate_reply

BRAND_ID = "AppleSupport"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
TAXONOMY_PATH = os.path.join(DATA_DIR, "intent_taxonomy.json")
SEED = 42
N_STYLE_EXEMPLARS = 3

# Benchmark definitions. `headline` is the one the README promises under 15
# minutes; `full` is the extended run. Both are deterministic given SEED.
BENCHMARKS = {
    "headline": {"n": 40,
                 "description": "40-example stratified subset of the 200-example golden set"},
    "full": {"n": None,
             "description": "all 200 golden-set examples"},
}
HEADLINE_BUDGET_SECONDS = 900


def ensure_taxonomy(threads):
    if os.path.exists(TAXONOMY_PATH):
        taxonomy = load_taxonomy(TAXONOMY_PATH)
        if taxonomy.get("intents"):
            return taxonomy
    print("Discovering intents from data...")
    taxonomy = discover_intents_from_data([t["customer_message"] for t in threads], BRAND_ID)
    save_taxonomy(taxonomy, TAXONOMY_PATH)
    return taxonomy


def build_retrieval_corpus(threads, golden_rows):
    """Remove every golden-set example from the retrieval corpus."""
    golden_messages = {g["customer_message"].strip() for g in golden_rows}
    corpus = [t for t in threads if t["customer_message"].strip() not in golden_messages]
    removed = len(threads) - len(corpus)
    print(f"Retrieval corpus: {len(corpus)} threads "
          f"({removed} golden threads held out to prevent retrieval leakage)")
    return corpus


def out_of_fold_intents(golden_rows, n_splits=4):
    """
    Cross-fitted intent predictions for both baselines.

    Fitting on the whole golden set and scoring on part of it reports a training
    score. Cross-fitting keeps every row evaluable with no row predicted by a
    model that saw its own label.
    """
    from sklearn.model_selection import StratifiedKFold

    messages = [g["customer_message"] for g in golden_rows]
    labels = [g["gold_intent"] for g in golden_rows]
    min_class = min(Counter(labels).values())
    n_splits = max(2, min(n_splits, min_class))
    print(f"Cross-fitting baselines with {n_splits} folds "
          f"(smallest intent class has {min_class} examples)")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    trivial_override, tfidf_override = {}, {}
    for train_idx, test_idx in skf.split(messages, labels):
        tr_msgs = [messages[i] for i in train_idx]
        tr_lbls = [labels[i] for i in train_idx]

        tfidf = TfidfClassifier(); tfidf.fit(tr_msgs, tr_lbls)
        trivial = TrivialClassifier(mode="majority"); trivial.fit(tr_msgs, tr_lbls)
        for i in test_idx:
            tfidf_override[messages[i]] = tfidf.predict(messages[i])
            trivial_override[messages[i]] = trivial.predict(messages[i])
    return trivial_override, tfidf_override


from src.rate_limiter import default_rate_limiter


def run_agent(agent, golden_rows, label, checkpoint_path=None, resume=True):
    """Run one agent over the benchmark, checkpointing after every example."""
    results = []
    if checkpoint_path and resume and os.path.exists(checkpoint_path):
        with open(checkpoint_path, encoding="utf-8") as f:
            cached = json.load(f)
        if len(cached) <= len(golden_rows) and all(
            c["customer_message"] == g["customer_message"]
            for c, g in zip(cached, golden_rows)
        ):
            results = cached
            if results:
                print(f"  [{label}] Resuming from checkpoint at "
                      f"{len(results)}/{len(golden_rows)}")
        else:
            print(f"  [{label}] Checkpoint does not match this benchmark; ignoring.")

    for i in range(len(results), len(golden_rows)):
        print(f"  [{label}] Processing {i+1}/{len(golden_rows)}...", end="\r", flush=True)
        try:
            res = agent.handle_message(golden_rows[i]["customer_message"])
        except Exception as e:
            print(f"\n  [{label}] Example {i+1} failed: {e}", flush=True)
            res = {
                "customer_message": golden_rows[i]["customer_message"],
                "intent": {"intent": "unknown", "confidence": 0.0, "failed": True},
                "escalation": {"should_escalate": False, "reason": f"error: {e}", "failed": True},
                "reply": "",
                "failed": True,
                "error": str(e),
                "processing_time": 0.0,
            }
        results.append(res)
        if checkpoint_path:
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump(results, f)
    print(f"  [{label}] Processed {len(golden_rows)} examples.          ", flush=True)
    return results


def build_failure_analysis(main_results, golden_rows, judgments):
    """
    Collect genuine failures with the context the report needs.

    Severity ranks safety first: a missed escalation can leave a legal threat or a
    safety hazard answered by a bot, which is categorically worse than a
    mislabelled intent.
    """
    rank = ["low", "medium", "high"]
    failures = []
    for i, (res, gold) in enumerate(zip(main_results, golden_rows)):
        pred_intent = res["intent"]["intent"]
        gold_intent = gold.get("gold_intent", "")
        pred_esc = bool(res["escalation"]["should_escalate"])
        gold_esc = bool(gold["gold_should_escalate"])

        reasons, severity = [], "low"
        if res.get("failed"):
            reasons.append("api_failure")
            severity = "high"
        if gold_esc and not pred_esc:
            reasons.append("escalation_false_negative")
            severity = "high"
        if pred_esc and not gold_esc:
            reasons.append("escalation_false_positive")
            severity = max(severity, "medium", key=rank.index)
        if gold_intent and pred_intent != gold_intent:
            reasons.append("intent_misclassification")
            severity = max(severity, "medium", key=rank.index)

        v = validate_reply(res.get("reply", ""), res.get("similar_conversations"))
        for viol in v["violations"]:
            reasons.append(f"reply_{viol['type']}")
            if viol["type"] in ("pii_exposed", "empty_reply"):
                severity = "high"

        judge_overall = None
        if judgments and i < len(judgments):
            judge_overall = judgments[i].get("overall_score")
            if judge_overall is not None and judge_overall <= 2.5:
                reasons.append("low_judge_score")

        if not reasons:
            continue

        failures.append({
            "example_id": gold.get("example_id"),
            "difficulty": gold.get("difficulty", ""),
            "customer_message": gold["customer_message"],
            "gold_intent": gold_intent,
            "predicted_intent": pred_intent,
            "gold_escalation": gold_esc,
            "predicted_escalation": pred_esc,
            "escalation_reason_given": res["escalation"].get("reason", ""),
            "generated_reply": res.get("reply", ""),
            "judge_overall": judge_overall,
            "failure_reasons": reasons,
            "severity": severity,
            "hypothesis": "",
            "mitigation": "",
        })

    order = {"high": 0, "medium": 1, "low": 2}
    failures.sort(key=lambda f: (order[f["severity"]], -len(f["failure_reasons"])))
    return failures


def main():
    ap = argparse.ArgumentParser(description="Run the evaluation benchmark")
    ap.add_argument("--benchmark", choices=sorted(BENCHMARKS), default="headline",
                    help="which benchmark definition to run")
    ap.add_argument("--no-judge", action="store_true", help="skip the LLM judge")
    ap.add_argument("--local", action="store_true", help="force local Ollama")
    ap.add_argument("--ollama-model", default="llama3.2:latest")
    ap.add_argument("--fresh", action="store_true", help="ignore checkpoints")
    ap.add_argument("--allow-prelabelled", action="store_true",
                    help="run against heuristic pre-labels; results stamped PROVISIONAL")
    ap.add_argument("--max-retries", type=int, default=int(os.environ.get("GROQ_MAX_RETRIES", 3)),
                    help="maximum retries for API 429 errors (default: 3)")
    ap.add_argument("--rate-limit-tpm", type=int, default=int(os.environ.get("GROQ_MAX_TPM", 7000)),
                    help="tokens-per-minute rate limit budget (default: 7000)")
    ap.add_argument("--min-request-interval", type=float,
                    default=float(os.environ.get("GROQ_MIN_REQUEST_INTERVAL", 2.0)),
                    help="minimum interval in seconds between requests (default: 2.0)")
    args = ap.parse_args()

    default_rate_limiter.configure(
        max_tpm=args.rate_limit_tpm,
        min_interval=args.min_request_interval,
        max_retries=args.max_retries,
    )

    bench = BENCHMARKS[args.benchmark]
    wall_start = time.time()

    print("=" * 64)
    print(f"AI Support Agent — benchmark: {args.benchmark}")
    print(f"  {bench['description']}")
    print("=" * 64)

    # ── Providers ────────────────────────────────────────────────────
    use_groq = check_groq() and not args.local
    ollama_ok = check_ollama()
    if not use_groq and not ollama_ok:
        print("NOTE: neither Groq nor local Ollama is reachable.")
        print("Running in local deterministic baseline mode for evaluation.")
    agent_model = DEFAULT_GROQ_MODEL if use_groq else f"ollama/{args.ollama_model}"
    judge_model = JUDGE_GROQ_MODEL if use_groq else f"ollama/{args.ollama_model}"
    print(f"Agent model: {agent_model}")
    print(f"Judge model: {judge_model}"
          + ("" if judge_model != agent_model else
             "   [!] same model as agent — self-preference bias possible"))
    print(f"Rate limiter: max_tpm={args.rate_limit_tpm}, min_interval={args.min_request_interval}s, max_retries={args.max_retries}")

    # ── Golden set with provenance enforcement ───────────────────────
    print("\n--- Golden set ---")
    try:
        all_rows, provenance = load_evaluation_set(
            allow_prelabelled=args.allow_prelabelled or args.local
        )
    except GoldenSetProvenanceError as e:
        print(f"\nREFUSING TO RUN:{e}")
        return 3
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        return 3

    if provenance["schema_problems"]:
        print(f"ERROR: golden set has {len(provenance['schema_problems'])} schema problems:")
        for p in provenance["schema_problems"][:10]:
            print(f"  - {p}")
        return 3

    print(f"Loaded {provenance['total']} rows from {provenance['path']} "
          f"(sha256:{provenance['sha256_16']})")
    print(f"  human_reviewed={provenance['human_reviewed']}  "
          f"pre_labelled={provenance['pre_labelled_heuristic']}  "
          f"pending={provenance['pending_review']}")
    if provenance["labels_are_provisional"]:
        print("\n  *** " + provenance["warning"] + " ***\n")

    n = bench["n"]
    golden_rows = (stratified_subsample(all_rows, n, seed=SEED) if n
                   else sorted(all_rows, key=lambda r: int(r["example_id"])))
    n_esc = sum(1 for g in golden_rows if g["gold_should_escalate"])
    full_esc = sum(1 for g in all_rows if g["gold_should_escalate"])
    print(f"Benchmark set: {len(golden_rows)} examples, escalation base rate "
          f"{n_esc}/{len(golden_rows)} = {n_esc/len(golden_rows):.1%} "
          f"(full set {full_esc}/{len(all_rows)} = {full_esc/len(all_rows):.1%})")

    # ── Data / taxonomy / corpus ─────────────────────────────────────
    threads = load_processed_data(BRAND_ID)
    taxonomy = ensure_taxonomy(threads)
    intent_names = [i["name"] for i in taxonomy.get("intents", [])]
    print(f"\nTaxonomy: {len(intent_names)} intents (sha256:{file_hash(TAXONOMY_PATH)})")

    # Hold out ALL 200 golden messages, not only the benchmark subset, so the
    # headline and full benchmarks share one corpus and remain comparable.
    corpus = build_retrieval_corpus(threads, all_rows)
    rng = random.Random(SEED)
    style_exemplars = [t["brand_reply"] for t in rng.sample(corpus, N_STYLE_EXEMPLARS)]

    print("\n--- Cross-fitting baselines ---")
    trivial_override, tfidf_override = out_of_fold_intents(all_rows)

    # ── Agents ───────────────────────────────────────────────────────
    print("\n--- Agents ---")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    agents = {
        "trivial": TrivialAgent(taxonomy,
                                train_labels=[g["gold_intent"] for g in all_rows],
                                intent_override=trivial_override),
        "simple": SimpleAgent(corpus, taxonomy, intent_override=tfidf_override),
        "main": SupportAgent(corpus, taxonomy, brand_name=BRAND_ID,
                             model=args.ollama_model, use_groq=use_groq),
    }

    # ── Run ──────────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("Running")
    print("=" * 64)
    runs, timings = {}, {}
    for name, agent in agents.items():
        print(f"\n--- {name.upper()} ---")
        t0 = time.time()
        runs[name] = run_agent(
            agent, golden_rows, name,
            checkpoint_path=os.path.join(
                RESULTS_DIR, f"{name}_raw_results_{args.benchmark}.json"),
            resume=not args.fresh,
        )
        timings[name] = round(time.time() - t0, 2)

    # ── Leakage gates ────────────────────────────────────────────────
    print("\n--- Leakage checks ---")
    try:
        leak_records = run_all_checks(
            corpus, golden_rows, runs,
            {"trivial": trivial_override, "simple": tfidf_override},
        )
    except LeakageError as e:
        print(f"\nLEAKAGE DETECTED — refusing to report results:\n  {e}")
        return 4
    for rec in leak_records:
        tag = f" [{rec['system']}]" if "system" in rec else ""
        print(f"  PASS  {rec['check']}{tag}")

    # ── Metrics ──────────────────────────────────────────────────────
    all_results = {}
    for name in agents:
        metrics = full_evaluation(runs[name], golden_rows, label_set=intent_names)
        metrics["wall_time_seconds"] = timings[name]
        metrics["reply_validation"] = validate_batch(runs[name], intent_names)
        all_results[name] = {"metrics": metrics}
        print(format_results_table(metrics, name=name.upper()))
        save_results(metrics, os.path.join(RESULTS_DIR, f"{name}_metrics.json"))

    # ── Judge ────────────────────────────────────────────────────────
    if not args.no_judge:
        print("\n" + "=" * 64)
        print(f"LLM-as-Judge  (model: {judge_model})")
        print("=" * 64)
        for name in agents:
            print(f"\n--- Judging {name.upper()} ---")
            judgments = judge_batch(
                runs[name], style_exemplars=style_exemplars, use_groq=use_groq,
                checkpoint_path=os.path.join(
                    RESULTS_DIR, f"{name}_judgments_raw_{args.benchmark}.json"),
                resume=not args.fresh, ollama_model=args.ollama_model,
            )
            agg = aggregate_judgments(judgments)
            agg["n_judged"] = len(judgments)
            agg["n_judge_errors"] = sum(1 for j in judgments if j.get("judge_error"))
            all_results[name]["judge"] = agg
            print(f"  overall {agg.get('overall', {}).get('mean', 'n/a')}/5 "
                  f"(n={agg['n_judged']}, errors={agg['n_judge_errors']})")
            save_results(agg, os.path.join(RESULTS_DIR, f"{name}_judge.json"))

    # ── Comparison ───────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("COMPARISON")
    print("=" * 64)
    print(f"{'Metric':<26}{'Trivial':>11}{'Simple':>11}{'Main':>11}")
    print("-" * 59)
    rowdefs = [
        ("Intent accuracy", lambda r: r["metrics"]["intent_classification"]["accuracy"]),
        ("Intent macro-F1", lambda r: r["metrics"]["intent_classification"]["macro_f1"]),
        ("Intent weighted-F1", lambda r: r["metrics"]["intent_classification"]["weighted_f1"]),
        ("Escalation precision", lambda r: r["metrics"]["escalation"]["precision"]),
        ("Escalation recall", lambda r: r["metrics"]["escalation"]["recall"]),
        ("Escalation F1", lambda r: r["metrics"]["escalation"]["f1"]),
        ("ROUGE-1", lambda r: r["metrics"]["rouge_scores"]["rouge1"]),
        ("ROUGE-2", lambda r: r["metrics"]["rouge_scores"]["rouge2"]),
        ("ROUGE-L", lambda r: r["metrics"]["rouge_scores"]["rougeL"]),
        ("Empty replies", lambda r: r["metrics"]["reply_stats"]["empty_replies"]),
        ("Reply violations", lambda r: r["metrics"]["reply_validation"]["n_replies_with_violation"]),
        ("Avg latency (s)", lambda r: r["metrics"]["performance"]["avg_time_seconds"]),
    ]
    if not args.no_judge:
        for dim in ["overall", "relevance", "groundedness", "helpfulness",
                    "tone", "completeness"]:
            rowdefs.append((f"Judge {dim}",
                            lambda r, d=dim: r.get("judge", {}).get(d, {}).get("mean", 0)))
    for label, get in rowdefs:
        vals = []
        for name in ("trivial", "simple", "main"):
            try:
                vals.append(f"{get(all_results[name]):.4f}")
            except (KeyError, TypeError):
                vals.append("N/A")
        print(f"{label:<26}{vals[0]:>11}{vals[1]:>11}{vals[2]:>11}")

    total_wall = round(time.time() - wall_start, 1)

    run_config = {
        "benchmark_name": args.benchmark,
        "benchmark_description": bench["description"],
        "benchmark_size": len(golden_rows),
        "golden_set_total": len(all_rows),
        "seed": SEED,
        "agent_model": agent_model,
        "judge_model": judge_model,
        "judge_differs_from_agent": judge_model != agent_model,
        "provider": "groq" if use_groq else "ollama",
        "temperature": {"intent": 0.1, "reply": 0.4, "escalation": 0.1, "judge": 0.1},
        "taxonomy_sha256_16": file_hash(TAXONOMY_PATH),
        "taxonomy_intents": intent_names,
        "golden_set_sha256_16": provenance["sha256_16"],
        "golden_set_path": provenance["path"],
        "golden_set_provenance": provenance,
        "labels_are_provisional": provenance["labels_are_provisional"],
        "retrieval_corpus_size": len(corpus),
        "golden_threads_held_out": len(threads) - len(corpus),
        "escalation_base_rate": round(n_esc / len(golden_rows), 4),
        "style_exemplar_count": N_STYLE_EXEMPLARS,
        "leakage_checks": leak_records,
        "total_wall_time_seconds": total_wall,
        "per_system_wall_time_seconds": timings,
        "judge_enabled": not args.no_judge,
    }

    comparison = {"run_config": run_config}
    for name in agents:
        comparison[name] = {"metrics": all_results[name]["metrics"],
                            "judge": all_results[name].get("judge", {})}
    if provenance["labels_are_provisional"]:
        comparison["PROVISIONAL"] = provenance["warning"]

    save_results(comparison, os.path.join(RESULTS_DIR, f"comparison_{args.benchmark}.json"))
    save_results(run_config, os.path.join(RESULTS_DIR, f"run_config_{args.benchmark}.json"))

    # ── Failure analysis ─────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("FAILURE ANALYSIS (main agent)")
    print("=" * 64)
    judgments = None
    jp = os.path.join(RESULTS_DIR, f"main_judgments_raw_{args.benchmark}.json")
    if os.path.exists(jp):
        with open(jp, encoding="utf-8") as f:
            judgments = json.load(f)

    failures = build_failure_analysis(runs["main"], golden_rows, judgments)
    by_reason = Counter(r for f in failures for r in f["failure_reasons"])
    by_severity = Counter(f["severity"] for f in failures)

    print(f"\nExamples with >=1 failure: {len(failures)}/{len(golden_rows)}")
    print(f"By severity: {dict(by_severity)}")
    print("By reason:")
    for r, c in by_reason.most_common():
        print(f"  {r}: {c}")
    if len(failures) < 5:
        print(f"\nNOTE: only {len(failures)} failing examples on this benchmark. "
              f"The report must say so rather than pad the list to five.")
    for i, f in enumerate(failures[:5], 1):
        print(f"\n  {i}. [{f['severity']}] id={f['example_id']} ({f['difficulty']})")
        print(f"     msg:    {f['customer_message'][:130]}")
        print(f"     intent: {f['predicted_intent']} (gold {f['gold_intent']})")
        print(f"     esc:    {f['predicted_escalation']} (gold {f['gold_escalation']})")
        print(f"     why:    {', '.join(f['failure_reasons'])}")

    save_results({
        "benchmark": args.benchmark,
        "labels_are_provisional": provenance["labels_are_provisional"],
        "summary": {
            "n_evaluated": len(golden_rows),
            "n_examples_with_failure": len(failures),
            "by_severity": dict(by_severity),
            "by_reason": dict(by_reason),
        },
        "failures": failures,
    }, os.path.join(RESULTS_DIR, f"failure_analysis_{args.benchmark}.json"))

    print(f"\n{'='*64}")
    print(f"Total wall time: {total_wall}s ({total_wall/60:.1f} min)")
    if args.benchmark == "headline":
        verdict = "WITHIN" if total_wall < HEADLINE_BUDGET_SECONDS else "OVER"
        print(f"Headline budget is 15 min ({HEADLINE_BUDGET_SECONDS}s): {verdict}")
    print(f"Artifacts written to results/ with suffix _{args.benchmark}")
    if provenance["labels_are_provisional"]:
        print("\n*** RESULTS ARE PROVISIONAL — scored against heuristic pre-labels. ***")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
