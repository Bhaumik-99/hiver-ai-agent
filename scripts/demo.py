"""
Single-message demo of the full pipeline.

    python scripts/demo.py "@AppleSupport my battery dies in 2 hours since iOS 11"
    python scripts/demo.py --local "..."
    echo "..." | python scripts/demo.py

Prints every stage — intent, retrieved historical precedent, drafted reply,
reply validation, and the escalation decision with its stated reason — so the
grounding is inspectable rather than implied.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.agent import SupportAgent
from src.data_processor import load_processed_data
from src.golden_set import PRELABELLED_PATH, read_rows
from src.intent_classifier import load_taxonomy
from src.ollama_client import DEFAULT_GROQ_MODEL, check_groq, check_ollama
from src.reply_validation import validate_reply

BRAND_ID = "AppleSupport"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("message", nargs="*", help="the customer message")
    ap.add_argument("--local", action="store_true", help="force local Ollama")
    ap.add_argument("--ollama-model", default="llama3.2:latest")
    args = ap.parse_args()

    message = " ".join(args.message).strip() or sys.stdin.read().strip()
    if not message:
        print("Provide a message as an argument or on stdin.")
        return 2

    use_groq = check_groq() and not args.local
    if not use_groq and not check_ollama():
        print("No LLM backend. Set GROQ_API_KEY in .env, or run Ollama and pass --local.")
        return 2

    threads = load_processed_data(BRAND_ID)
    # Hold out the golden set here too, so the demo shows the same grounding the
    # benchmark measures rather than a privileged view of the evaluation data.
    golden = {r["customer_message"].strip() for r in read_rows(PRELABELLED_PATH)}
    corpus = [t for t in threads if t["customer_message"].strip() not in golden]

    agent = SupportAgent(corpus, load_taxonomy(), brand_name=BRAND_ID,
                         model=args.ollama_model, use_groq=use_groq)
    result = agent.handle_message(message)

    bar = "─" * 68
    print(f"\n{bar}\nCUSTOMER\n{bar}\n{message}")

    intent = result["intent"]
    print(f"\n{bar}\nINTENT\n{bar}")
    print(f"  {intent['intent']}   (confidence {intent.get('confidence', 'n/a')})")
    if intent.get("reasoning"):
        print(f"  reasoning: {intent['reasoning']}")
    if intent.get("fallback"):
        print("  [!] fallback used — the model did not return a valid taxonomy intent")

    print(f"\n{bar}\nRETRIEVED PRECEDENT (what the reply is grounded in)\n{bar}")
    for i, conv in enumerate(result.get("similar_conversations", []), 1):
        print(f"  {i}. similarity {conv['similarity']}")
        print(f"     customer: {conv['customer_message'][:110]}")
        print(f"     brand:    {conv['brand_reply'][:110]}")
    if not result.get("similar_conversations"):
        print("  (no sufficiently similar historical thread found)")

    reply = result["reply"]
    print(f"\n{bar}\nDRAFTED REPLY\n{bar}\n{reply or '(empty — generation failed)'}")

    v = validate_reply(reply, result.get("similar_conversations"))
    print(f"\n  length {v['length']}/280   valid: {v['valid']}")
    for viol in v["violations"]:
        print(f"  [violation] {viol['type']}: {viol['detail']}")

    esc = result["escalation"]
    print(f"\n{bar}\nROUTING DECISION\n{bar}")
    print(f"  {'ESCALATE TO HUMAN' if esc['should_escalate'] else 'AUTO-HANDLE'}")
    print(f"  reason:     {esc.get('reason', '')}")
    print(f"  decided by: {esc.get('method', '')}   "
          f"confidence: {esc.get('confidence', 'n/a')}")

    print(f"\n{bar}")
    print(f"  model {DEFAULT_GROQ_MODEL if use_groq else 'ollama/' + args.ollama_model}"
          f"   corpus {len(corpus)} threads   {result['processing_time']}s")
    print(bar)
    return 0


if __name__ == "__main__":
    sys.exit(main())
