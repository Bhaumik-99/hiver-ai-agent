"""
Validation of generated reply tweets.

Every check here corresponds to something that would be a real incident if the
agent posted it to a live @AppleSupport handle. Violations are COUNTED and
reported, never suppressed: the report may only claim zero violations of a class
when the benchmark actually observed zero.
"""

import re

TWITTER_CHAR_LIMIT = 280

# Support hosts the brand actually uses in this dataset. A generated reply that
# invents a plausible-looking apple.com deep link is worse than one that offers
# no link, because the customer will try to follow it.
ALLOWED_URL_HOSTS = {
    "t.co", "apple.com", "www.apple.com", "support.apple.com",
    "iforgot.apple.com", "appleid.apple.com", "getsupport.apple.com",
    "discussions.apple.com",
}

URL_PATTERN = re.compile(r'https?://([^\s/]+)(/\S*)?', re.IGNORECASE)

# Personally identifying data that must never appear in a public reply.
PII_PATTERNS = {
    "email": re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'),
    "phone": re.compile(r'(?<!\w)(?:\+?\d{1,2}[\s.-])?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}(?!\w)'),
    "credit_card": re.compile(r'(?<!\d)(?:\d{4}[\s-]){3}\d{4}(?!\d)'),
    "serial_like": re.compile(r'(?<!\w)[A-Z0-9]{2}[A-Z]{3}[A-Z0-9]{5,7}(?!\w)'),
}

# Text that indicates the model echoed its own scaffolding into the reply.
CONTEXT_LEAK_MARKERS = [
    "example 1:", "example 2:", "example 3:",
    "customer:", "detected intent:", "similar conversation",
    "here are examples", "based on the examples", "as an ai",
    "system prompt", "reply tweet:", "brand_reply",
]


def _find_urls(text: str):
    return [(m.group(1).lower(), m.group(0)) for m in URL_PATTERN.finditer(text)]


def validate_reply(reply: str, retrieved_context: list[dict] = None) -> dict:
    """
    Validate one generated reply.

    Returns {"valid": bool, "violations": [ {type, detail} ]}.
    """
    violations = []
    text = reply or ""

    if not text.strip():
        violations.append({"type": "empty_reply",
                           "detail": "generator returned no text"})
        # Nothing further is meaningful for an empty reply.
        return {"valid": False, "violations": violations,
                "length": 0, "n_violations": len(violations)}

    if len(text) > TWITTER_CHAR_LIMIT:
        violations.append({
            "type": "char_limit_exceeded",
            "detail": f"{len(text)} chars > {TWITTER_CHAR_LIMIT}",
        })

    lowered = text.lower()
    for marker in CONTEXT_LEAK_MARKERS:
        if marker in lowered:
            violations.append({
                "type": "context_leak",
                "detail": f"reply contains scaffolding marker {marker!r}",
            })
            break

    # Verbatim echo of a retrieved neighbour's text is also a context leak: the
    # reply should be grounded in the examples, not a copy of them.
    if retrieved_context:
        for conv in retrieved_context[:3]:
            src = (conv.get("customer_message") or "").strip()
            if len(src) > 40 and src.lower() in lowered:
                violations.append({
                    "type": "context_leak",
                    "detail": "reply reproduces a retrieved customer message verbatim",
                })
                break

    for kind, pattern in PII_PATTERNS.items():
        m = pattern.search(text)
        if m:
            violations.append({
                "type": "pii_exposed",
                "detail": f"{kind} pattern present in a public reply",
            })

    for host, full in _find_urls(text):
        host_clean = host.split(":")[0]
        if host_clean not in ALLOWED_URL_HOSTS:
            violations.append({
                "type": "invalid_support_url",
                "detail": f"link to non-brand host {host_clean!r}",
            })
        elif host_clean == "t.co" and not re.match(r'^https?://t\.co/\w{5,}$', full):
            violations.append({
                "type": "invalid_support_url",
                "detail": f"malformed t.co short link {full!r}",
            })

    return {
        "valid": not violations,
        "violations": violations,
        "length": len(text),
        "n_violations": len(violations),
    }


def validate_intent_output(intent: str, taxonomy_names: list[str]) -> dict:
    """Check a predicted intent is inside the declared taxonomy."""
    ok = intent in taxonomy_names
    return {
        "valid": ok,
        "violations": ([] if ok else
                       [{"type": "invalid_intent",
                         "detail": f"{intent!r} is not in the taxonomy"}]),
    }


def validate_batch(agent_results: list[dict], taxonomy_names: list[str]) -> dict:
    """
    Validate every reply and intent in a benchmark run.

    Returns aggregate counts plus concrete examples, so the report can quote the
    number of violations actually observed rather than asserting there were none.
    """
    counts: dict[str, int] = {}
    examples: dict[str, list] = {}
    n_invalid_replies = 0

    for i, r in enumerate(agent_results):
        res = validate_reply(r.get("reply", ""), r.get("similar_conversations"))
        if not res["valid"]:
            n_invalid_replies += 1
        for v in res["violations"]:
            counts[v["type"]] = counts.get(v["type"], 0) + 1
            examples.setdefault(v["type"], []).append({
                "index": i,
                "customer_message": (r.get("customer_message") or "")[:120],
                "reply": (r.get("reply") or "")[:160],
                "detail": v["detail"],
            })

        iv = validate_intent_output(r.get("intent", {}).get("intent", ""), taxonomy_names)
        for v in iv["violations"]:
            counts[v["type"]] = counts.get(v["type"], 0) + 1
            examples.setdefault(v["type"], []).append({
                "index": i, "detail": v["detail"],
            })

    n = len(agent_results)
    lengths = [len(r.get("reply") or "") for r in agent_results]
    return {
        "n_replies": n,
        "n_replies_with_violation": n_invalid_replies,
        "violation_rate": round(n_invalid_replies / n, 4) if n else 0.0,
        "counts_by_type": counts,
        "examples_by_type": {k: v[:5] for k, v in examples.items()},
        "max_reply_length": max(lengths) if lengths else 0,
        "n_over_char_limit": sum(1 for L in lengths if L > TWITTER_CHAR_LIMIT),
    }
