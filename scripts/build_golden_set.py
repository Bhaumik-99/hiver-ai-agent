"""
Build the golden evaluation set via stratified sampling.
Samples ~200 examples from processed threads and creates a CSV template for hand-labelling.
"""
import sys
import os
import json
import csv
import random
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

BRAND_ID = "AppleSupport"
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
GOLDEN_SET_PATH = os.path.join(DATA_DIR, "golden_set.csv")
TARGET_SIZE = 200


def load_threads():
    path = os.path.join(PROCESSED_DIR, f"{BRAND_ID}_threads.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def classify_difficulty(msg: str) -> str:
    """Heuristic difficulty classification for stratification."""
    msg_lower = msg.lower()
    
    if any(w in msg_lower for w in ["fuck", "shit", "damn", "wtf", "lawsuit", "lawyer"]):
        return "edge_case"
    
    if len(msg.split()) <= 5:
        return "hard_short"
    
    question_marks = msg.count("?")
    if question_marks >= 2 or len(msg) > 200:
        return "hard_complex"
    
    return "standard"


def sample_golden_set(threads: list[dict], target: int = TARGET_SIZE) -> list[dict]:
    random.seed(42)
    
    buckets = {"standard": [], "hard_short": [], "hard_complex": [], "edge_case": []}
    for t in threads:
        difficulty = classify_difficulty(t["customer_message"])
        buckets[difficulty].append(t)
    
    for k, v in buckets.items():
        print(f"  {k}: {len(v)}")
    
    allocations = {
        "standard": int(target * 0.30),
        "hard_short": int(target * 0.25),
        "hard_complex": int(target * 0.25),
        "edge_case": int(target * 0.20),
    }
    
    sampled = []
    for bucket_name, count in allocations.items():
        available = buckets[bucket_name]
        n = min(count, len(available))
        sampled.extend(random.sample(available, n))
    
    remaining = target - len(sampled)
    if remaining > 0:
        all_remaining = [t for t in threads if t not in sampled]
        sampled.extend(random.sample(all_remaining, min(remaining, len(all_remaining))))
    
    random.shuffle(sampled)
    return sampled[:target]


def create_golden_csv(sampled: list[dict], output_path: str = GOLDEN_SET_PATH):
    """Create CSV template for hand-labelling."""
    fieldnames = [
        "example_id",
        "customer_message",
        "brand_reply",
        "num_turns",
        "difficulty",
        # ─── Labels to fill in ───
        "gold_intent",
        "gold_should_escalate",
        "gold_escalation_reason",
        "gold_reply_quality",  # 1-5: how good is the brand's actual reply
        "labelling_notes",
    ]
    
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for i, t in enumerate(sampled):
            msg = t["customer_message"]
            writer.writerow({
                "example_id": i + 1,
                "customer_message": msg,
                "brand_reply": t["brand_reply"],
                "num_turns": t.get("num_turns", 2),
                "difficulty": classify_difficulty(msg),
                "gold_intent": "",
                "gold_should_escalate": "",
                "gold_escalation_reason": "",
                "gold_reply_quality": "",
                "labelling_notes": "",
            })
    
    print(f"Created golden set template with {len(sampled)} examples at {output_path}")


def auto_label_golden_set(input_path: str = GOLDEN_SET_PATH,
                           output_path: str = None,
                           taxonomy: dict = None):
    from src.ollama_client import generate_json
    
    if output_path is None:
        output_path = input_path.replace(".csv", "_labelled.csv")
    
    if taxonomy is None:
        tax_path = os.path.join(DATA_DIR, "intent_taxonomy.json")
        if os.path.exists(tax_path):
            with open(tax_path) as f:
                taxonomy = json.load(f)
    
    intent_names = [i["name"] for i in taxonomy.get("intents", [])] if taxonomy else []
    intent_desc = "\n".join(
        f"- {i['name']}: {i['description']}"
        for i in taxonomy.get("intents", [])
    ) if taxonomy else "No taxonomy loaded"
    
    rows = []
    with open(input_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    
    print(f"Auto-labelling {len(rows)} examples...")
    
    for i, row in enumerate(rows):
        print(f"  Labelling {i+1}/{len(rows)}...", end="\r")
        msg = row["customer_message"]
        
        prompt = f"""Analyze this customer support message sent to AppleSupport.

Message: "{msg[:300]}"

Available intent categories:
{intent_desc}

Respond with JSON:
{{
    "intent": "<one of the intent names>",
    "should_escalate": true/false,
    "escalation_reason": "<reason if escalating, otherwise empty>",
    "reply_quality_of_brand": <1-5 score for how good Apple's actual reply is>
}}

Apple's actual reply was: "{row.get('brand_reply', '')[:200]}"
"""
        
        result = generate_json(prompt, temperature=0.1, max_tokens=200)
        
        if "parse_error" not in result:
            row["gold_intent"] = result.get("intent", "")
            row["gold_should_escalate"] = str(result.get("should_escalate", False)).lower()
            row["gold_escalation_reason"] = result.get("escalation_reason", "")
            row["gold_reply_quality"] = str(result.get("reply_quality_of_brand", 3))
            row["labelling_notes"] = "auto-labelled (needs human review)"
    
    # Write output
    fieldnames = list(rows[0].keys())
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"\nAuto-labelled golden set saved to {output_path}")
    return output_path


if __name__ == "__main__":
    threads = load_threads()
    sampled = sample_golden_set(threads)
    create_golden_csv(sampled)
    
    # Print difficulty distribution
    difficulties = [classify_difficulty(t["customer_message"]) for t in sampled]
    from collections import Counter
    print("\nDifficulty distribution:")
    for k, v in Counter(difficulties).most_common():
        print(f"  {k}: {v}")
