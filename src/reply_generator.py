"""
Reply generator that drafts brand-appropriate responses.
Supports three approaches:
1. Trivial: echo template
2. Simple: verbatim nearest-neighbor retrieval
3. Main: LLM generation grounded in retrieved historical responses
"""

from src.ollama_client import generate
from src.retriever import ConversationRetriever


class TrivialReplyGenerator:
    """Baseline 1: Always returns a generic template."""

    TEMPLATES = {
        "default": "Thank you for reaching out! We'd like to help. Could you please DM us your details so we can look into this further?",
        "account_access": "We're sorry you're having trouble accessing your account. Please try resetting your password at https://iforgot.apple.com.",
        "technical_issue": "We understand this is frustrating. Could you try restarting your device? If the issue persists, DM us the details.",
    }

    def generate(self, message: str, intent: str = "default",
                 similar_convos: list = None) -> str:
        return self.TEMPLATES.get(intent, self.TEMPLATES["default"])


class NearestNeighborReplyGenerator:
    """Baseline 2: Returns the brand reply from the most similar historical conversation."""

    def __init__(self, retriever: ConversationRetriever):
        self.retriever = retriever

    def generate(self, message: str, intent: str = None,
                 similar_convos: list = None) -> str:
        if similar_convos and len(similar_convos) > 0:
            return similar_convos[0]["brand_reply"]

        results = self.retriever.retrieve(message, top_k=1)
        if results:
            return results[0]["brand_reply"]
        return "Thank you for reaching out. Please DM us so we can assist you further."


class LLMReplyGenerator:
    """
    Main system: LLM generates a reply grounded in retrieved historical responses.
    The key insight is that we show the LLM how Apple *actually* responds to similar
    issues, constraining it to stay in-character.
    """

    def __init__(self, brand_name: str = "AppleSupport", model: str = "llama3.2", use_groq: bool = False):
        self.brand_name = brand_name
        self.model = model
        self.use_groq = use_groq

    def generate(self, message: str, intent: str = None,
                 similar_convos: list = None) -> str:
        context_block = ""
        if similar_convos:
            examples = []
            for i, conv in enumerate(similar_convos[:2], 1):
                examples.append(
                    f"Example {i}:\n"
                    f"  Customer: {conv['customer_message'][:140]}\n"
                    f"  Reply: {conv['brand_reply'][:140]}"
                )
            context_block = "\n".join(examples)

        intent_note = f" (Intent: {intent})" if intent else ""

        system_prompt = (
            f"You are {self.brand_name} on Twitter. Write an empathetic, actionable reply under "
            "280 characters starting with @customer and directing to DM for account-specific issues."
        )

        prompt = f"""Customer: "{message[:200]}"{intent_note}

Reference examples:
{context_block}

Write a single reply tweet matching brand voice:"""

        try:
            reply = generate(
                prompt,
                model=self.model,
                temperature=0.4,
                max_tokens=100,
                system=system_prompt,
                use_groq=self.use_groq,
            )
        except Exception as e:
            print(f"\n  [generation failed] {e}", flush=True)
            return ""

        reply = reply.strip()
        if reply.startswith('"') and reply.endswith('"'):
            reply = reply[1:-1]
        for prefix in ["Reply:", "Response:", "Tweet:", f"{self.brand_name}:"]:
            if reply.startswith(prefix):
                reply = reply[len(prefix):].strip()

        return reply
