"""
Render the benchmark artifacts into the markdown tables used by README.md and
REPORT.md.

Every number in the documentation is generated from results/comparison_*.json by
this script. Nothing is typed by hand, so a documented figure cannot drift away
from the run that produced it — the previous version of this report quoted an
intent accuracy of 82.5% that appeared in no results file at all.

    python scripts/render_results.py --benchmark headline
    python scripts/render_results.py --benchmark headline --check   # verify docs match
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE, "results")

BEGIN = "<!-- BEGIN GENERATED RESULTS: {benchmark} -->"
END = "<!-- END GENERATED RESULTS: {benchmark} -->"


def _fmt(v, pct=False, dp=3):
    if v is None:
        return "n/a"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, (int,)) and not pct:
        return str(v)
    try:
        return f"{v:.1%}" if pct else f"{v:.{dp}f}"
    except (TypeError, ValueError):
        return str(v)


def render(benchmark: str) -> str:
    path = os.path.join(RESULTS_DIR, f"comparison_{benchmark}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run: python scripts/run_evaluation.py "
            f"--benchmark {benchmark}")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    cfg = data["run_config"]
    systems = [s for s in ("trivial", "simple", "main") if s in data]
    L = []

    provisional = cfg.get("labels_are_provisional")
    if provisional:
        L += [
            "> **These numbers are PROVISIONAL.** They were scored against the",
            "> heuristic pre-labels in `data/golden_set_prelabelled.csv`, not against",
            "> human annotations. They measure agreement with",
            "> `scripts/label_golden_set.py`, not with a human annotator. See",
            "> `SUBMISSION_AUDIT.md`; the golden set is marked PARTIAL until the",
            "> annotation workflow is completed by a person.",
            "",
        ]

    L += [
        f"**Benchmark `{cfg['benchmark_name']}`** — {cfg['benchmark_description']}.",
        "",
        f"| Run parameter | Value |",
        f"|---|---|",
        f"| Examples scored | {cfg['benchmark_size']} of {cfg['golden_set_total']} golden examples |",
        f"| Escalation base rate | {_fmt(cfg['escalation_base_rate'], pct=True)} |",
        f"| Agent model | `{cfg['agent_model']}` |",
        f"| Judge model | `{cfg['judge_model']}` |",
        f"| Judge differs from agent | {_fmt(cfg['judge_differs_from_agent'])} |",
        f"| Provider | {cfg['provider']} |",
        f"| Seed | {cfg['seed']} |",
        f"| Retrieval corpus | {cfg['retrieval_corpus_size']} threads "
        f"({cfg['golden_threads_held_out']} golden threads held out) |",
        f"| Golden set | `{cfg['golden_set_path']}` sha256:`{cfg['golden_set_sha256_16']}` |",
        f"| Taxonomy | {len(cfg['taxonomy_intents'])} intents, sha256:`{cfg['taxonomy_sha256_16']}` |",
        f"| Label provenance | "
        f"{'heuristic pre-labels (PROVISIONAL)' if provisional else 'human-reviewed'} |",
        f"| Total wall time | {cfg['total_wall_time_seconds']}s "
        f"({cfg['total_wall_time_seconds']/60:.1f} min) |",
        "",
    ]

    hdr = "| Metric | " + " | ".join(s.capitalize() for s in systems) + " |"
    sep = "|---|" + "|".join([":---:"] * len(systems)) + "|"

    def row(label, get, pct=False, dp=3):
        cells = []
        for s in systems:
            try:
                cells.append(_fmt(get(data[s]), pct=pct, dp=dp))
            except (KeyError, TypeError, IndexError):
                cells.append("n/a")
        return f"| {label} | " + " | ".join(cells) + " |"

    L += ["#### Intent classification", "", hdr, sep,
          row("Accuracy", lambda d: d["metrics"]["intent_classification"]["accuracy"], pct=True),
          row("Accuracy 95% CI",
              lambda d: (f"{d['metrics']['intent_classification']['accuracy_ci95']['lo']:.1%}"
                         f"–{d['metrics']['intent_classification']['accuracy_ci95']['hi']:.1%}")),
          row("Macro-F1", lambda d: d["metrics"]["intent_classification"]["macro_f1"]),
          row("Weighted-F1", lambda d: d["metrics"]["intent_classification"]["weighted_f1"]),
          row("Off-taxonomy predictions",
              lambda d: d["metrics"]["intent_classification"]["n_off_taxonomy"]),
          ""]

    L += ["#### Escalation", "", hdr, sep,
          row("Accuracy", lambda d: d["metrics"]["escalation"]["accuracy"], pct=True),
          row("Accuracy 95% CI",
              lambda d: (f"{d['metrics']['escalation']['accuracy_ci95']['lo']:.1%}"
                         f"–{d['metrics']['escalation']['accuracy_ci95']['hi']:.1%}")),
          row("Precision", lambda d: d["metrics"]["escalation"]["precision"], pct=True),
          row("Recall", lambda d: d["metrics"]["escalation"]["recall"], pct=True),
          row("F1", lambda d: d["metrics"]["escalation"]["f1"]),
          row("TP / FP / FN / TN",
              lambda d: "{tp} / {fp} / {fn} / {tn}".format(**d["metrics"]["escalation"]["confusion"])),
          row("False negatives (missed handoffs)",
              lambda d: d["metrics"]["escalation"]["confusion"]["fn"]),
          ""]

    L += ["#### Reply quality", "", hdr, sep,
          row("ROUGE-1", lambda d: d["metrics"]["rouge_scores"]["rouge1"], dp=4),
          row("ROUGE-2", lambda d: d["metrics"]["rouge_scores"]["rouge2"], dp=4),
          row("ROUGE-L", lambda d: d["metrics"]["rouge_scores"]["rougeL"], dp=4),
          row("Mean reply length (chars)",
              lambda d: d["metrics"]["reply_stats"]["avg_length_chars"], dp=1),
          ""]

    L += ["#### Output validation (observed violations)", "", hdr, sep,
          row("Replies with >=1 violation",
              lambda d: d["metrics"]["reply_validation"]["n_replies_with_violation"]),
          row("Empty replies", lambda d: d["metrics"]["reply_stats"]["empty_replies"]),
          row("Over 280 chars",
              lambda d: d["metrics"]["reply_validation"]["n_over_char_limit"]),
          row("Longest reply (chars)",
              lambda d: d["metrics"]["reply_validation"]["max_reply_length"]),
          ""]

    viol_types = sorted({t for s in systems
                         for t in data[s]["metrics"]["reply_validation"]["counts_by_type"]})
    if viol_types:
        L += ["Violation counts by type:", "", hdr, sep]
        for t in viol_types:
            L.append(row(f"`{t}`",
                         lambda d, t=t: d["metrics"]["reply_validation"]["counts_by_type"].get(t, 0)))
        L.append("")
    else:
        L += ["No output-validation violations were observed on this benchmark "
              "for any system.", ""]

    if cfg.get("judge_enabled") and any(data[s].get("judge") for s in systems):
        L += [f"#### LLM-as-judge (1–5), judge model `{cfg['judge_model']}`", "",
              hdr, sep]
        for dim in ("overall", "relevance", "groundedness", "helpfulness",
                    "tone", "completeness"):
            L.append(row(dim.capitalize(),
                         lambda d, k=dim: d["judge"][k]["mean"], dp=2))
        L.append(row("Std dev (overall)", lambda d: d["judge"]["overall"]["std"], dp=2))
        L.append(row("Replies judged", lambda d: d["judge"]["n_judged"]))
        L.append(row("Judge errors", lambda d: d["judge"]["n_judge_errors"]))
        L.append("")

    L += ["#### Latency (observed, includes provider rate-limit waiting)", "",
          hdr, sep,
          row("Mean seconds / message",
              lambda d: d["metrics"]["performance"]["avg_time_seconds"], dp=2),
          ""]

    # Leakage gate results
    checks = cfg.get("leakage_checks", [])
    if checks:
        names = []
        for c in checks:
            n = c["check"] + (f" [{c['system']}]" if "system" in c else "")
            names.append(n)
        L += [f"All {len(checks)} leakage assertions passed for this run "
              f"(the harness aborts instead of reporting a leaked number): "
              + ", ".join(f"`{n}`" for n in sorted(set(names))) + ".", ""]

    # Judge calibration status
    cal_path = os.path.join(RESULTS_DIR, "judge_calibration.json")
    if os.path.exists(cal_path):
        with open(cal_path, encoding="utf-8") as f:
            cal = json.load(f)
        if cal.get("status") == "COMPUTED":
            L += [f"#### Judge vs human agreement (n={cal['n_samples']})", "",
                  "| Dimension | Spearman | Exact | Adjacent ±1 | MAE | QWK |",
                  "|---|:---:|:---:|:---:|:---:|:---:|"]
            for dim, m in cal["per_dimension"].items():
                qwk = m.get("quadratic_weighted_kappa")
                qwk_txt = "n/a" if qwk is None else f"{qwk:.3f}"
                L.append(f"| {dim} | {m['spearman_correlation']:.3f} | "
                         f"{m['exact_agreement']:.1%} | {m['adjacent_agreement']:.1%} | "
                         f"{m['mean_absolute_error']:.2f} | {qwk_txt} |")
            L.append("")
        else:
            L += ["#### Judge vs human agreement", "",
                  f"**PENDING** — {cal.get('reason', 'not yet measured')}. "
                  "No agreement statistic is estimated in its place.", ""]
    else:
        L += ["#### Judge vs human agreement", "",
              "**PENDING** — run `python scripts/compute_judge_calibration.py` "
              "after a human scores `data/judge_calibration/judge_calibration_task.csv`.",
              ""]

    # Inter-annotator agreement status
    ann_path = os.path.join(RESULTS_DIR, "annotation_agreement.json")
    if os.path.exists(ann_path):
        with open(ann_path, encoding="utf-8") as f:
            ann = json.load(f)
        if ann.get("status") == "COMPUTED":
            L += [f"#### Inter-annotator agreement (n={ann['n_overlap_items']} overlap rows)", "",
                  "| Decision | Cohen's κ | Raw agreement | Disagreements |",
                  "|---|:---:|:---:|:---:|"]
            for dim in ("intent", "escalation"):
                d = ann[dim]
                k = d["cohens_kappa"]
                L.append(f"| {dim} | {'undefined' if k is None else f'{k:.3f}'} | "
                         f"{d['raw_agreement']:.1%} | {d['n_disagreements']}/{d['n_items']} |")
            L.append("")
        else:
            L += ["#### Inter-annotator agreement", "",
                  f"**PENDING** — {ann.get('reason', 'not yet measured')}", ""]

    L += [f"*Generated by `scripts/render_results.py --benchmark {benchmark}` from "
          f"`results/comparison_{benchmark}.json`. Do not edit by hand.*"]
    return "\n".join(L)


def inject(doc_path: str, benchmark: str, block: str) -> bool:
    begin, end = BEGIN.format(benchmark=benchmark), END.format(benchmark=benchmark)
    if not os.path.exists(doc_path):
        return False
    with open(doc_path, encoding="utf-8") as f:
        text = f.read()
    if begin not in text or end not in text:
        return False
    new = re.sub(re.escape(begin) + r".*?" + re.escape(end),
                 f"{begin}\n{block}\n{end}", text, flags=re.DOTALL)
    if new != text:
        with open(doc_path, "w", encoding="utf-8") as f:
            f.write(new)
        return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="headline")
    ap.add_argument("--check", action="store_true",
                    help="exit non-zero if the docs are out of date")
    args = ap.parse_args()

    block = render(args.benchmark)
    targets = [os.path.join(BASE, "README.md"), os.path.join(BASE, "REPORT.md")]

    if args.check:
        stale = []
        for t in targets:
            if not os.path.exists(t):
                continue
            with open(t, encoding="utf-8") as f:
                text = f.read()
            begin = BEGIN.format(benchmark=args.benchmark)
            if begin in text and block not in text:
                stale.append(os.path.basename(t))
        if stale:
            print(f"OUT OF DATE: {', '.join(stale)} do not match "
                  f"results/comparison_{args.benchmark}.json")
            print(f"Fix with: python scripts/render_results.py --benchmark {args.benchmark}")
            return 1
        print(f"Documentation matches results/comparison_{args.benchmark}.json")
        return 0

    for t in targets:
        if inject(t, args.benchmark, block):
            print(f"Updated {os.path.basename(t)}")
        else:
            print(f"No marker block for '{args.benchmark}' in {os.path.basename(t)} "
                  f"(skipped)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
