"""
Full evaluation pipeline — runs all three systems on the golden set and produces comparison.

Usage:
    python scripts/run_evaluation.py                  # Full run (all 200 golden examples)
    python scripts/run_evaluation.py --n 40           # Stratified subsample of 40
    python scripts/run_evaluation.py --no-judge       # Skip LLM judge
    python scripts/run_evaluation.py --no-calibration # Skip judge calibration pass

Evaluation-integrity invariants enforced here (see REPORT.md "What is misleading"):
  1. Golden examples are removed from the retrieval corpus, so no system can
     retrieve the reference reply it is being scored against.
  2. Baseline intent classifiers are scored on OUT-OF-FOLD predictions, so they
     are never evaluated on examples they were fitted on.
  3. All systems are scored over the same explicit taxonomy label set, so
     macro-F1 denominators are identical.
  4. The judge sees a fixed, system-independent set of brand-voice exemplars,
     never the reference reply and never a system's own retrieved context.
"""
import sys
import os
import json
import csv
import argparse
import random
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from src.data_processor import load_processed_data
from src.intent_classifier import (
    LLMClassifier, TfidfClassifier, TrivialClassifier,
    load_taxonomy, discover_intents_from_data, save_taxonomy,
)
from src.agent import SupportAgent, TrivialAgent, SimpleAgent
from src.evaluator import full_evaluation, format_results_table, save_results
from src.llm_judge import judge_batch, aggregate_judgments, run_judge_calibration
from src.ollama_client import check_ollama, check_groq, DEFAULT_GROQ_MODEL
from src.llm_judge import JUDGE_GROQ_MODEL

BRAND_ID = "AppleSupport"
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
SEED = 42
N_STYLE_EXEMPLARS = 3


def load_golden_set(path: str = None) -> list[dict]:
    """Load the labelled golden set."""
    if path is None:
        path = os.path.join(DATA_DIR, "golden_set_labelled.csv")
        if not os.path.exists(path):
            path = os.path.join(DATA_DIR, "golden_set.csv")

    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            escalate = row.get("gold_should_escalate", "false").strip().lower()
            row["gold_should_escalate"] = escalate in ("true", "1", "yes")

            try:
                row["gold_reply_quality"] = int(row.get("gold_reply_quality", 3))
            except (ValueError, TypeError):
                row["gold_reply_quality"] = 3

            rows.append(row)

    print(f"Loaded {len(rows)} golden set examples from {path}")
    return rows


def ensure_taxonomy(threads: list[dict]) -> dict:
    """Load or create intent taxonomy."""
    tax_path = os.path.join(DATA_DIR, "intent_taxonomy.json")

    if os.path.exists(tax_path):
        taxonomy = load_taxonomy(tax_path)
        if taxonomy.get("intents"):
            print(f"Loaded existing taxonomy with {len(taxonomy['intents'])} intents")
            return taxonomy

    print("Discovering intents from data...")
    messages = [t["customer_message"] for t in threads]
    taxonomy = discover_intents_from_data(messages, BRAND_ID)
    save_taxonomy(taxonomy, tax_path)
    return taxonomy


def build_retrieval_corpus(threads: list[dict], golden_set: list[dict]) -> list[dict]:
    """
    Remove every golden-set example from the retrieval corpus.

    The golden set was sampled FROM these threads, so without this the top
    neighbour of a golden message is its own thread and the nearest-neighbour
    baseline returns the reference reply verbatim. That is what produced the
    previous run's ROUGE-1 of 0.797 and groundedness of 5.0/5.0 with zero
    variance for the "simple" baseline — a measurement of the leak, not the system.
    """
    golden_messages = {g["customer_message"] for g in golden_set}
    corpus = [t for t in threads if t["customer_message"] not in golden_messages]
    removed = len(threads) - len(corpus)
    print(f"Retrieval corpus: {len(corpus)} threads "
          f"({removed} golden-set threads held out to prevent retrieval leakage)")
    if removed < len(golden_messages):
        print(f"  NOTE: {len(golden_messages) - removed} golden messages were not "
              f"found in the corpus (deduplication or text normalisation).")
    return corpus


def stratified_subsample(golden_set: list[dict], n: int, seed: int = SEED) -> list[dict]:
    """
    Take a subsample that preserves the escalation base rate.

    Slicing `golden_set[:n]` instead gave the previous run a 10% escalation base
    rate against the full set's 34%, so the escalation metrics were describing a
    different distribution from the one claimed.
    """
    if n >= len(golden_set):
        return golden_set
    rng = random.Random(seed)
    pos = [g for g in golden_set if g["gold_should_escalate"]]
    neg = [g for g in golden_set if not g["gold_should_escalate"]]
    n_pos = round(n * len(pos) / len(golden_set))
    n_pos = max(1, min(n_pos, len(pos), n - 1))
    sample = rng.sample(pos, n_pos) + rng.sample(neg, min(n - n_pos, len(neg)))
    rng.shuffle(sample)
    return sample


def out_of_fold_intents(golden_set: list[dict], n_splits: int = 4) -> tuple[dict, dict]:
    """
    Produce out-of-fold intent predictions for both baselines.

    The previous version fitted the TF-IDF classifier on the ENTIRE golden set and
    then evaluated it on a slice of that same set, so its reported accuracy was a
    training score. Cross-fitting keeps every one of the 200 examples evaluable
    while guaranteeing no example is predicted by a model that saw its label.

    Returns (trivial_override, tfidf_override), each message -> intent result dict.
    """
    from sklearn.model_selection import StratifiedKFold
    from collections import Counter

    messages = [g["customer_message"] for g in golden_set]
    labels = [g["gold_intent"] for g in golden_set]

    # StratifiedKFold needs at least n_splits members in every class.
    min_class = min(Counter(labels).values())
    n_splits = max(2, min(n_splits, min_class))
    print(f"Cross-fitting baselines with {n_splits} folds "
          f"(smallest intent class has {min_class} examples)")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    trivial_override, tfidf_override = {}, {}

    for train_idx, test_idx in skf.split(messages, labels):
        train_msgs = [messages[i] for i in train_idx]
        train_lbls = [labels[i] for i in train_idx]

        tfidf = TfidfClassifier()
        tfidf.fit(train_msgs, train_lbls)
        trivial = TrivialClassifier(mode="majority")
        trivial.fit(train_msgs, train_lbls)

        for i in test_idx:
            tfidf_override[messages[i]] = tfidf.predict(messages[i])
            trivial_override[messages[i]] = trivial.predict(messages[i])

    return trivial_override, tfidf_override


def run_agent_on_golden_set(agent, golden_set: list[dict],
                            label: str = "Agent",
                            checkpoint_path: str = None,
                            resume: bool = True) -> list[dict]:
    """
    Run an agent on the golden set and collect results.

    Checkpoints after every example. A full main-agent pass is ~600 sequential
    LLM round trips; losing all of it to one transient API error — and then
    re-spending the rate-limited quota to redo it — is not an acceptable
    failure mode for a harness meant to be re-run.
    """
    results = []
    if checkpoint_path and resume and os.path.exists(checkpoint_path):
        with open(checkpoint_path, encoding="utf-8") as f:
            cached = json.load(f)
        # Only reuse a checkpoint that lines up with the current golden set.
        if len(cached) <= len(golden_set) and all(
            c["customer_message"] == g["customer_message"]
            for c, g in zip(cached, golden_set)
        ):
            results = cached
            if results:
                print(f"  [{label}] Resuming from checkpoint at "
                      f"{len(results)}/{len(golden_set)}")
        else:
            print(f"  [{label}] Checkpoint does not match current golden set; ignoring.")

    total = len(golden_set)
    for i in range(len(results), total):
        print(f"  [{label}] Processing {i+1}/{total}...", end="\r", flush=True)
        results.append(agent.handle_message(golden_set[i]["customer_message"]))
        if checkpoint_path:
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump(results, f)

    print(f"  [{label}] Processed {total} examples.          ", flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description="Run full evaluation pipeline")
    parser.add_argument("--no-judge", action="store_true", help="Skip LLM-as-judge")
    parser.add_argument("--no-calibration", action="store_true",
                        help="Skip the judge calibration pass")
    parser.add_argument("--use-groq", action="store_true", help="Force Groq API")
    parser.add_argument("--local", action="store_true",
                        help="Force local Ollama even when a Groq key is present")
    parser.add_argument("--ollama-model", default="llama3.2:latest",
                        help="Ollama model to use for the agent and judge")
    parser.add_argument("--fresh", action="store_true",
                        help="Ignore checkpoints and re-run every agent from scratch")
    parser.add_argument("--n", type=int, default=None,
                        help="Evaluate on a stratified subsample of this size "
                             "(default: the full golden set)")
    parser.add_argument("--calibration-n", type=int, default=60,
                        help="Examples used for the judge calibration pass")
    args = parser.parse_args()

    print("=" * 60)
    print("Hiver AI Support Agent — Evaluation Pipeline")
    print("=" * 60)

    groq_available = check_groq() and not args.local
    use_groq = (args.use_groq or groq_available) and not args.local
    print(f"[{'OK' if use_groq else 'SKIP'}] Groq API: {'enabled' if use_groq else 'disabled'}")

    if not use_groq and not check_ollama():
        print("ERROR: Neither Groq nor Ollama is available. Start Ollama or provide GROQ_API_KEY in .env")
        sys.exit(1)
    if not use_groq:
        print(f"[OK] Ollama is running — agent and judge both on {args.ollama_model}")
        print("     NOTE: with a single local model the judge and the generator are the")
        print("     SAME model, which invites self-preference bias. Recorded in run_config.")

    print("\n--- Loading data ---")
    threads = load_processed_data(BRAND_ID)
    print(f"Loaded {len(threads)} conversation threads")

    print("\n--- Taxonomy ---")
    taxonomy = ensure_taxonomy(threads)
    intent_names = [i["name"] for i in taxonomy.get("intents", [])]
    print(f"Intents ({len(intent_names)}): {', '.join(intent_names)}")

    print("\n--- Golden Set ---")
    full_golden_set = load_golden_set()
    golden_set = (stratified_subsample(full_golden_set, args.n)
                  if args.n else full_golden_set)
    n_esc = sum(1 for g in golden_set if g["gold_should_escalate"])
    print(f"Evaluating on {len(golden_set)}/{len(full_golden_set)} examples "
          f"(escalation base rate {n_esc}/{len(golden_set)} = {n_esc/len(golden_set):.1%})")

    print("\n--- Leakage control ---")
    corpus = build_retrieval_corpus(threads, full_golden_set)

    rng = random.Random(SEED)
    style_exemplars = [t["brand_reply"] for t in
                       rng.sample(corpus, N_STYLE_EXEMPLARS)]
    print(f"Judge will see {len(style_exemplars)} fixed brand-voice exemplars "
          f"(identical for all systems)")

    print("\n--- Cross-fitting baselines ---")
    trivial_override, tfidf_override = out_of_fold_intents(full_golden_set)

    print("\n--- Initializing agents ---")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("1. Trivial agent (majority intent / template reply / never escalate)...")
    trivial_agent = TrivialAgent(
        taxonomy,
        train_labels=[g["gold_intent"] for g in full_golden_set],
        intent_override=trivial_override,
    )

    print("2. Simple agent (TF-IDF intent + nearest-neighbour reply + rules)...")
    simple_agent = SimpleAgent(corpus, taxonomy, intent_override=tfidf_override)

    print(f"3. Main agent (LLM + grounded retrieval, Groq={use_groq})...")
    main_agent = SupportAgent(corpus, taxonomy, brand_name=BRAND_ID,
                              model=args.ollama_model, use_groq=use_groq)

    print("\n" + "=" * 60)
    print("Running Evaluation")
    print("=" * 60)

    all_results = {}

    for name, agent in [("trivial", trivial_agent), ("simple", simple_agent),
                        ("main", main_agent)]:
        print(f"\n--- {name.upper()} AGENT ---")
        start = time.time()
        results = run_agent_on_golden_set(
            agent, golden_set, label=name,
            checkpoint_path=os.path.join(RESULTS_DIR, f"{name}_raw_results.json"),
            resume=not args.fresh,
        )
        elapsed = time.time() - start

        metrics = full_evaluation(results, golden_set, label_set=intent_names)
        metrics["wall_time"] = round(elapsed, 2)

        all_results[name] = {"metrics": metrics, "results": results}

        print(format_results_table(metrics, name=name.upper()))
        save_results(metrics, os.path.join(RESULTS_DIR, f"{name}_metrics.json"))

    if not args.no_judge:
        print("\n" + "=" * 60)
        print("LLM-as-Judge Evaluation")
        print("=" * 60)
        print(f"Judge model: {JUDGE_GROQ_MODEL if groq_available else 'ollama local'} "
              f"(agent model: {DEFAULT_GROQ_MODEL if use_groq else 'ollama/llama3.2'})")

        for name in ["trivial", "simple", "main"]:
            print(f"\n--- Judging {name.upper()} agent replies ---")
            judgments = judge_batch(
                all_results[name]["results"],
                style_exemplars=style_exemplars,
                use_groq=groq_available,
                checkpoint_path=os.path.join(RESULTS_DIR, f"{name}_judgments_raw.json"),
                resume=not args.fresh,
                ollama_model=args.ollama_model,
            )

            agg = aggregate_judgments(judgments)
            agg["n_judge_errors"] = sum(1 for j in judgments if j.get("judge_error"))
            all_results[name]["judge"] = agg

            print(f"  Overall: {agg.get('overall', {}).get('mean', 'N/A')}/5")
            for dim in ["relevance", "groundedness", "helpfulness", "tone", "completeness"]:
                if dim in agg:
                    print(f"  {dim}: {agg[dim]['mean']}/5 (std={agg[dim]['std']})")

            save_results(agg, os.path.join(RESULTS_DIR, f"{name}_judge.json"))
            save_results(judgments, os.path.join(RESULTS_DIR, f"{name}_judgments_raw.json"))

        if not args.no_calibration:
            print("\n" + "=" * 60)
            print("Judge Calibration (judge vs. reference reply-quality labels)")
            print("=" * 60)
            calib = run_judge_calibration(
                golden_set,
                style_exemplars=style_exemplars,
                use_groq=groq_available,
                n=args.calibration_n,
                ollama_model=args.ollama_model,
            )
            print(f"  Spearman rho:       {calib['spearman_correlation']} "
                  f"(p={calib['spearman_p_value']})")
            print(f"  Exact agreement:    {calib['exact_agreement']:.1%}")
            print(f"  Adjacent (+/-1):    {calib['adjacent_agreement']:.1%}")
            print(f"  Mean abs error:     {calib['mean_absolute_error']}")
            print(f"  Label source:       {calib['label_source']}")
            save_results(calib, os.path.join(RESULTS_DIR, "judge_calibration.json"))
            all_results["judge_calibration"] = calib

    print("\n" + "=" * 60)
    print("COMPARISON TABLE")
    print("=" * 60)

    print(f"{'Metric':<25} {'Trivial':>10} {'Simple':>10} {'Main':>10}")
    print("-" * 57)

    metrics_to_compare = [
        ("Intent Accuracy", lambda r: r["metrics"]["intent_classification"]["accuracy"]),
        ("Intent Macro-F1", lambda r: r["metrics"]["intent_classification"]["macro_f1"]),
        ("Escalation Precision", lambda r: r["metrics"]["escalation"]["precision"]),
        ("Escalation Recall", lambda r: r["metrics"]["escalation"]["recall"]),
        ("Escalation F1", lambda r: r["metrics"]["escalation"]["f1"]),
        ("ROUGE-1", lambda r: r["metrics"]["rouge_scores"]["rouge1"]),
        ("ROUGE-L", lambda r: r["metrics"]["rouge_scores"]["rougeL"]),
        ("Avg Time (s)", lambda r: r["metrics"]["performance"]["avg_time_seconds"]),
    ]

    if not args.no_judge:
        metrics_to_compare.extend([
            ("Judge Overall", lambda r: r.get("judge", {}).get("overall", {}).get("mean", 0)),
            ("Judge Relevance", lambda r: r.get("judge", {}).get("relevance", {}).get("mean", 0)),
            ("Judge Groundedness", lambda r: r.get("judge", {}).get("groundedness", {}).get("mean", 0)),
            ("Judge Helpfulness", lambda r: r.get("judge", {}).get("helpfulness", {}).get("mean", 0)),
            ("Judge Tone", lambda r: r.get("judge", {}).get("tone", {}).get("mean", 0)),
        ])

    for metric_name, getter in metrics_to_compare:
        vals = []
        for name in ["trivial", "simple", "main"]:
            try:
                vals.append(f"{getter(all_results[name]):.4f}")
            except (KeyError, TypeError):
                vals.append("N/A")
        print(f"{metric_name:<25} {vals[0]:>10} {vals[1]:>10} {vals[2]:>10}")

    comparison = {
        "run_config": {
            "n_evaluated": len(golden_set),
            "n_golden_total": len(full_golden_set),
            "escalation_base_rate": round(n_esc / len(golden_set), 4),
            "retrieval_corpus_size": len(corpus),
            "golden_threads_held_out": len(threads) - len(corpus),
            "label_set": intent_names,
            "seed": SEED,
            "agent_model": DEFAULT_GROQ_MODEL if use_groq else f"ollama/{args.ollama_model}",
            "judge_model": JUDGE_GROQ_MODEL if groq_available else f"ollama/{args.ollama_model}",
            "judge_model_differs_from_agent": (
                JUDGE_GROQ_MODEL != DEFAULT_GROQ_MODEL if groq_available else False),
            "baselines_cross_fitted": True,
            "judge_blind_to_reference": True,
        },
    }
    for name in ["trivial", "simple", "main"]:
        comparison[name] = {
            "metrics": all_results[name]["metrics"],
            "judge": all_results[name].get("judge", {}),
        }
    if "judge_calibration" in all_results:
        comparison["judge_calibration"] = {
            k: v for k, v in all_results["judge_calibration"].items() if k != "examples"
        }
    save_results(comparison, os.path.join(RESULTS_DIR, "comparison.json"))

    print("\n" + "=" * 60)
    print("FAILURE ANALYSIS (Main Agent)")
    print("=" * 60)

    main_results = all_results["main"]["results"]
    main_judgments = None
    judge_path = os.path.join(RESULTS_DIR, "main_judgments_raw.json")
    if not args.no_judge and os.path.exists(judge_path):
        with open(judge_path, encoding="utf-8") as f:
            main_judgments = json.load(f)

    failures = []
    for i, (result, gold) in enumerate(zip(main_results, golden_set)):
        pred_intent = result["intent"]["intent"]
        gold_intent = gold.get("gold_intent", "")
        pred_esc = result["escalation"]["should_escalate"]
        gold_esc = gold["gold_should_escalate"]

        issues = []
        if gold_intent and pred_intent != gold_intent:
            issues.append({
                "type": "intent",
                "detail": f"predicted '{pred_intent}' vs gold '{gold_intent}'",
            })
        if pred_esc and not gold_esc:
            issues.append({"type": "escalation_false_positive",
                           "detail": result["escalation"].get("reason", "")})
        if gold_esc and not pred_esc:
            issues.append({"type": "escalation_false_negative",
                           "detail": f"gold reason: {gold.get('gold_escalation_reason', '')}"})

        judge_overall = None
        if main_judgments and i < len(main_judgments):
            judge_overall = main_judgments[i].get("overall_score")
            if judge_overall is not None and judge_overall <= 2.5:
                issues.append({"type": "low_judge_score",
                               "detail": f"judge overall {judge_overall}/5"})

        if issues:
            failures.append({
                "example_index": i,
                "difficulty": gold.get("difficulty", ""),
                "message": result["customer_message"][:200],
                "gold_intent": gold_intent,
                "predicted_intent": pred_intent,
                "issues": issues,
                "reply": result["reply"][:200],
                "judge_overall": judge_overall,
            })

    # Group failures by type so the report can quote real frequencies.
    from collections import Counter
    type_counts = Counter(iss["type"] for f in failures for iss in f["issues"])
    print(f"\nExamples with at least one failure: {len(failures)} / {len(main_results)}")
    print("Failure counts by type:")
    for t, c in type_counts.most_common():
        print(f"  {t}: {c}")

    print("\nTop 5 failures:")
    for i, f in enumerate(failures[:5], 1):
        print(f"\n  {i}. [{f['difficulty']}] {f['message'][:140]}")
        for issue in f["issues"]:
            print(f"     -> {issue['type']}: {issue['detail']}")
        print(f"     Reply: {f['reply'][:140]}")

    save_results({
        "summary": {
            "n_evaluated": len(main_results),
            "n_examples_with_failure": len(failures),
            "counts_by_type": dict(type_counts),
        },
        "failures": failures,
    }, os.path.join(RESULTS_DIR, "failure_analysis.json"))

    print(f"\n{'='*60}")
    print(f"All results saved to {RESULTS_DIR}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
