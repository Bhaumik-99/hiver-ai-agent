"""
Escalation decision module.
Decides whether a customer message should be auto-handled or escalated to a human.
Uses a hybrid approach: hard rules + LLM soft-signal analysis.
"""

import re
from src.ollama_client import generate_json



# Every entry must be escalation-worthy on its own. Bare high-frequency words were
# removed after they fired on ordinary support traffic: "news" matched "any news on
# the update?", "hurt" matched everyday complaints, "fire"/"burn"/"shock" matched
# figurative use, and "court"/"sue" matched unrelated substrings.
ESCALATION_KEYWORDS = {
    "legal": ["lawyer", "lawsuit", "attorney", "legal action", "small claims",
              "sue you", "sue apple", "take you to court", "consumer rights"],
    "safety": ["caught fire", "catching fire", "exploded", "explode", "burned my",
               "burnt my", "electric shock", "shocked me", "swollen battery",
               "battery is swelling", "smoking", "melted"],
    "account_security": ["hacked", "compromised", "unauthorized charge",
                         "unauthorised charge", "identity theft", "fraud",
                         "someone else is using", "was stolen", "got stolen",
                         "stolen my", "stolen device", "stolen phone"],
    "severe_frustration": ["worst experience", "never buying", "never again",
                           "class action", "switching to android", "done with apple"],
    "media_threat": ["going to the media", "call the news", "contact the press",
                     "journalist", "make this go viral", "expose you"],
    # Money movement is never auto-handled: a refund, chargeback or duplicate
    # charge needs an agent with account access, and getting it wrong in public
    # is expensive.
    "billing_dispute": ["refund", "charged twice", "double charged", "chargeback",
                        "billed twice", "money back", "cancel my subscription",
                        "charged me for", "wrong amount"],
    # A customer explicitly asking for a person is the least ambiguous handoff
    # signal there is; no classifier should be second-guessing it.
    "explicit_human_request": ["speak to a human", "talk to a human", "real person",
                               "speak to someone", "talk to someone", "a manager",
                               "your supervisor", "a representative", "escalate this"],
}

# Strong profanity only. Mild expletives ("damn", "crap", "wtf") are ordinary
# register in this dataset; escalating on them buries genuine cases in noise.
PROFANITY_PATTERN = re.compile(
    r'\b(f+u+c+k+\w*|s+h+i+t+\w*|a+s+s+h+o+l+e|b+u+l+l+s+h+i+t+|stfu|c+u+n+t+)\b',
    re.IGNORECASE
)

PII_PATTERN = re.compile(
    r'(\b\d{3}[-.]?\d{3}[-.]?\d{4}\b)|'   # Phone numbers
    r'(\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b)|'  # Email
    r'(\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b)',  # Credit card
    re.IGNORECASE
)


def check_hard_rules(message: str) -> dict:
    """
    Check deterministic escalation rules.
    Returns {"should_escalate": bool, "reason": str, "rule": str} or None.
    """
    msg_lower = message.lower()

    # Check keyword categories
    for category, keywords in ESCALATION_KEYWORDS.items():
        for kw in keywords:
            if kw in msg_lower:
                return {
                    "should_escalate": True,
                    "reason": f"Message contains {category} indicator: '{kw}'",
                    "rule": category,
                    "confidence": 0.95,
                }

    # Check profanity
    if PROFANITY_PATTERN.search(message):
        return {
            "should_escalate": True,
            "reason": "Message contains strong profanity — customer is very frustrated",
            "rule": "profanity",
            "confidence": 0.9,
        }

    # Check PII (customer sharing sensitive info publicly)
    if PII_PATTERN.search(message):
        return {
            "should_escalate": True,
            "reason": "Message contains potential PII — needs immediate human attention to protect customer data",
            "rule": "pii_detected",
            "confidence": 0.95,
        }

    return None


def check_soft_signals(message: str, intent: str = None,
                       conversation_history: list = None,
                       use_groq: bool = False) -> dict:
    """
    Use LLM to assess soft escalation signals that rules can't catch.
    """
    history_context = ""
    if conversation_history:
        history_context = "\nPrevious messages in this conversation:\n"
        for msg in conversation_history[-3:]:
            role = "Customer" if msg.get("inbound") else "Brand"
            history_context += f"  {role}: {msg.get('text', '')[:150]}\n"

    intent_context = f" | Intent: {intent}" if intent else ""

    prompt = f"""Decide if this customer support query requires human escalation (account/billing action, legal, safety, high frustration).
Message: "{message[:200]}"{intent_context}

Respond with JSON: {{"should_escalate": true/false, "reason": "<brief>"}}"""

    result = generate_json(prompt, temperature=0.1, max_tokens=60, use_groq=use_groq)

    if result.get("failed"):
        return {
            "should_escalate": False,
            "reason": "api_failure",
            "confidence": 0.0,
            "failed": True,
        }

    if "parse_error" in result:
        return {
            "should_escalate": False,
            "reason": "Standard query suitable for automated resolution",
            "confidence": 0.5,
        }

    return result


def decide_escalation(message: str, intent: str = None,
                      conversation_history: list = None,
                      use_llm: bool = True,
                      use_groq: bool = False) -> dict:
    """
    Main escalation decision function.
    Combines hard rules (fast, deterministic) with LLM analysis (nuanced).
    
    Returns:
    {
        "should_escalate": bool,
        "reason": str,
        "confidence": float,
        "method": "hard_rule" | "llm_analysis" | "auto_handle"
    }
    """
    hard_result = check_hard_rules(message)
    if hard_result and hard_result["should_escalate"]:
        hard_result["method"] = "hard_rule"
        return hard_result

    if use_llm:
        soft_result = check_soft_signals(message, intent, conversation_history, use_groq=use_groq)
        if soft_result.get("should_escalate", False):
            soft_result["method"] = "llm_analysis"
            return soft_result

    return {
        "should_escalate": False,
        "reason": "Standard query suitable for automated resolution",
        "confidence": 0.7,
        "method": "auto_handle",
    }


class TrivialEscalation:
    """Baseline: never escalate."""

    def decide(self, message: str, intent: str = None,
               conversation_history: list = None) -> dict:
        return {
            "should_escalate": False,
            "reason": "Trivial baseline: never escalates",
            "confidence": 1.0,
            "method": "trivial_baseline",
        }


class RuleOnlyEscalation:
    """Simple baseline: only hard rules, no LLM."""

    def decide(self, message: str, intent: str = None,
               conversation_history: list = None) -> dict:
        result = check_hard_rules(message)
        if result:
            result["method"] = "rule_only_baseline"
            return result
        return {
            "should_escalate": False,
            "reason": "No rule triggered",
            "confidence": 0.7,
            "method": "rule_only_baseline",
        }
