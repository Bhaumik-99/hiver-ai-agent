"""
LLM-as-Judge evaluation for reply quality.
Uses a stronger model (Groq Llama 3.1 70B or local Ollama) to judge
the quality of generated replies across multiple dimensions.
"""

import json
import os
import numpy as np
from src.ollama_client import generate_json, check_groq

# The judge deliberately runs on a DIFFERENT model family from the agent
# (see src/ollama_client.DEFAULT_GROQ_MODEL). A model scoring its own output
# rates it systematically higher than a neutral grader would, and the prior
# setup used one model as both generator and judge.
JUDGE_GROQ_MODEL = os.environ.get("JUDGE_GROQ_MODEL", "openai/gpt-oss-120b")


# Evaluation rubric with 5 dimensions
RUBRIC = {
    "relevance": {
        "description": "Does the reply directly address the customer's specific issue?",
        "scoring": {
            1: "Completely off-topic or generic, ignores the customer's problem",
            2: "Tangentially related but misses the core issue",
            3: "Addresses the issue but incompletely or vaguely",
            4: "Addresses the issue well with minor gaps",
            5: "Directly and precisely addresses the customer's specific problem",
        }
    },
    "groundedness": {
        "description": "Is the reply consistent with how this brand actually responds? Does it avoid hallucination?",
        "scoring": {
            1: "Contains fabricated information or contradicts brand practices",
            2: "Mostly generic, doesn't reflect brand voice or actual practices",
            3: "Somewhat consistent with brand practices",
            4: "Consistent with brand voice, minor stylistic deviations",
            5: "Perfectly matches brand tone, style, and actual response patterns",
        }
    },
    "helpfulness": {
        "description": "Does the reply move toward resolution? Does it provide actionable next steps?",
        "scoring": {
            1: "Provides no useful information or next steps",
            2: "Acknowledges the issue but offers no actionable help",
            3: "Provides some guidance but lacks specificity",
            4: "Offers clear next steps with minor gaps",
            5: "Provides complete, actionable resolution guidance",
        }
    },
    "tone": {
        "description": "Is the tone appropriate? Empathetic, professional, brand-aligned?",
        "scoring": {
            1: "Rude, dismissive, or highly inappropriate",
            2: "Cold or robotic, lacks empathy",
            3: "Neutral, neither warm nor cold",
            4: "Warm and professional, minor tone issues",
            5: "Perfectly empathetic, professional, and brand-aligned",
        }
    },
    "completeness": {
        "description": "Does the reply cover all aspects of the customer's query?",
        "scoring": {
            1: "Ignores most aspects of the query",
            2: "Addresses only one aspect, ignores others",
            3: "Covers main aspects but misses important details",
            4: "Covers most aspects with minor omissions",
            5: "Comprehensively addresses all aspects of the query",
        }
    },
}


def judge_reply(customer_message: str, generated_reply: str,
                style_exemplars: list[str] = None,
                use_groq: bool = None, ollama_model: str = "llama3.2:latest") -> dict:
    """
    Use LLM to judge the quality of a generated reply.

    The judge is deliberately BLIND to the reference reply for this example.
    Showing it the ground-truth reply turns the judge into a similarity metric:
    a system that copies a historical reply verbatim then scores a perfect 5 on
    "groundedness" for having reproduced the very text it was shown, which is
    exactly the artifact that made the nearest-neighbour baseline look grounded.

    `style_exemplars` are a FIXED, system-independent sample of this brand's
    historical replies, identical for every system under evaluation. They give
    the judge a sense of brand voice without leaking this example's answer, and
    without handing any one system its own retrieved context as the yardstick.
    """
    if use_groq is None:
        use_groq = check_groq()

    rubric_text = ""
    for dim, info in RUBRIC.items():
        rubric_text += f"\n{dim.upper()}: {info['description']}\n"
        for score, desc in info["scoring"].items():
            rubric_text += f"  {score}: {desc}\n"

    context = ""
    if style_exemplars:
        context += "\n\nFor brand-voice reference, here is how this brand writes in general."
        context += "\nThese are UNRELATED to the message above — do not treat them as the expected answer:"
        for i, r in enumerate(style_exemplars[:3], 1):
            context += f"\n  Example {i}: \"{r[:200]}\""

    prompt = f"""You are evaluating the quality of an AI-generated customer support reply.

CUSTOMER MESSAGE: "{customer_message[:300]}"

AI-GENERATED REPLY: "{generated_reply[:300]}"
{context}

EVALUATION RUBRIC:
{rubric_text}

Score the AI-generated reply on each dimension (1-5). Think step by step.

Respond with JSON:
{{
    "relevance": {{"score": <1-5>, "reason": "<brief>"}},
    "groundedness": {{"score": <1-5>, "reason": "<brief>"}},
    "helpfulness": {{"score": <1-5>, "reason": "<brief>"}},
    "tone": {{"score": <1-5>, "reason": "<brief>"}},
    "completeness": {{"score": <1-5>, "reason": "<brief>"}},
    "overall_score": <1-5 average>,
    "overall_assessment": "<1-2 sentence summary>"
}}"""

    result = generate_json(prompt, model=ollama_model, temperature=0.1,
                           max_tokens=700, use_groq=use_groq,
                           groq_model=JUDGE_GROQ_MODEL)

    if "parse_error" in result:
        return {
            dim: {"score": 3, "reason": "Judge failed to parse"}
            for dim in RUBRIC
        } | {"overall_score": 3.0, "overall_assessment": "Judge parse error", "judge_error": True}

    for dim in RUBRIC:
        if dim in result:
            if isinstance(result[dim], dict):
                score = result[dim].get("score", 3)
                result[dim]["score"] = max(1, min(5, int(score)))
            elif isinstance(result[dim], (int, float)):
                result[dim] = {"score": max(1, min(5, int(result[dim]))), "reason": ""}

    scores = []
    for dim in RUBRIC:
        if dim in result and isinstance(result[dim], dict):
            scores.append(result[dim].get("score", 3))
    if scores:
        result["overall_score"] = round(sum(scores) / len(scores), 2)

    return result


def judge_batch(agent_results: list[dict], style_exemplars: list[str] = None,
                use_groq: bool = None, checkpoint_path: str = None,
                resume: bool = True, ollama_model: str = "llama3.2:latest") -> list[dict]:
    """
    Judge a batch of agent results.

    `style_exemplars` must be the SAME fixed list for every system being compared.
    Previously each system was judged against its own retrieved neighbours, which
    gave the nearest-neighbour baseline its own output as the reference standard.

    Checkpoints after every judgment for the same reason the agent runs do: this
    is one rate-limited API call per reply, per system.
    Returns list of judge scores for each example.
    """
    judgments = []
    if checkpoint_path and resume and os.path.exists(checkpoint_path):
        with open(checkpoint_path, encoding="utf-8") as f:
            cached = json.load(f)
        if isinstance(cached, list) and len(cached) <= len(agent_results):
            judgments = cached
            if judgments:
                print(f"  Resuming judge from checkpoint at "
                      f"{len(judgments)}/{len(agent_results)}")

    for i in range(len(judgments), len(agent_results)):
        print(f"  Judging {i+1}/{len(agent_results)}...", end="\r", flush=True)
        result = agent_results[i]

        try:
            judgment = judge_reply(
                customer_message=result["customer_message"],
                generated_reply=result["reply"],
                style_exemplars=style_exemplars,
                use_groq=use_groq,
                ollama_model=ollama_model,
            )
        except RuntimeError as e:
            # A judge call that never returns is a missing measurement, not a
            # score. Record it so `n_judge_errors` reports it instead of a
            # silent 3/5 quietly pulling every mean toward the middle.
            print(f"\n  [judge failed on example {i}] {e}")
            judgment = {"judge_error": True, "error": str(e)}

        judgment["example_index"] = i
        judgments.append(judgment)
        if checkpoint_path:
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump(judgments, f)

    print(f"  Judged {len(judgments)} examples.         ")
    return judgments


def aggregate_judgments(judgments: list[dict]) -> dict:
    """Aggregate judge scores across all examples."""
    dim_scores = {dim: [] for dim in RUBRIC}
    overall_scores = []

    for j in judgments:
        if j.get("judge_error"):
            continue
        for dim in RUBRIC:
            if dim in j and isinstance(j[dim], dict):
                dim_scores[dim].append(j[dim].get("score", 3))
        overall_scores.append(j.get("overall_score", 3))

    aggregate = {}
    for dim, scores in dim_scores.items():
        if scores:
            aggregate[dim] = {
                "mean": round(float(np.mean(scores)), 2),
                "std": round(float(np.std(scores)), 2),
                "min": int(min(scores)),
                "max": int(max(scores)),
                "n": len(scores),
            }

    if overall_scores:
        aggregate["overall"] = {
            "mean": round(float(np.mean(overall_scores)), 2),
            "std": round(float(np.std(overall_scores)), 2),
        }

    return aggregate


def compute_judge_human_agreement(judge_scores: list[float],
                                   human_scores: list[float]) -> dict:
    """
    Compute agreement between LLM judge and human scores.
    Reports Spearman correlation and exact/adjacent agreement rates.
    """
    from scipy import stats as scipy_stats

    n = min(len(judge_scores), len(human_scores))
    j = judge_scores[:n]
    h = human_scores[:n]

    # Spearman rank correlation
    if len(set(j)) > 1 and len(set(h)) > 1:
        spearman_r, spearman_p = scipy_stats.spearmanr(j, h)
    else:
        spearman_r, spearman_p = 0.0, 1.0

    # Exact agreement
    exact = sum(1 for a, b in zip(j, h) if a == b) / n if n > 0 else 0

    # Adjacent agreement (within 1 point)
    adjacent = sum(1 for a, b in zip(j, h) if abs(a - b) <= 1) / n if n > 0 else 0

    # Mean absolute error
    mae = float(np.mean([abs(a - b) for a, b in zip(j, h)])) if n > 0 else 0

    return {
        "spearman_correlation": round(float(spearman_r), 4),
        "spearman_p_value": round(float(spearman_p), 4),
        "exact_agreement": round(exact, 4),
        "adjacent_agreement": round(adjacent, 4),
        "mean_absolute_error": round(mae, 4),
        "n_samples": n,
    }
