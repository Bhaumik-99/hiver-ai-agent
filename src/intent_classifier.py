"""
Intent classifier for customer support messages.
Supports three approaches:
1. Trivial baseline: random/majority class
2. Simple baseline: TF-IDF + Logistic Regression
3. Main system: LLM few-shot classification
"""

import json
import os
import random
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from collections import Counter

from src.ollama_client import generate_json, generate


# Default intent taxonomy — will be refined from data
DEFAULT_TAXONOMY = {}


def load_taxonomy(path: str = None) -> dict:
    """Load intent taxonomy from JSON file."""
    if path is None:
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                            "data", "intent_taxonomy.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return DEFAULT_TAXONOMY


def discover_intents_from_data(messages: list[str], brand_name: str,
                               n_sample: int = 150) -> dict:
    """
    Use LLM to discover intent categories from a sample of customer messages.
    This is a data-driven approach — the taxonomy emerges from the data.
    """
    # Sample messages for discovery
    sample = random.sample(messages, min(n_sample, len(messages)))

    # Format messages for the LLM
    msg_block = "\n".join(f"{i+1}. {m[:200]}" for i, m in enumerate(sample[:50]))

    prompt = f"""You are analyzing customer support messages sent to {brand_name} on Twitter.

Here are 50 real customer messages:

{msg_block}

Based on these messages, identify 8-12 distinct intent categories that cover the majority of customer issues.

For each intent, provide:
- A short name (2-4 words, snake_case)
- A description (1 sentence)
- 2-3 example keywords or phrases

Respond with a JSON object like:
{{
    "intents": [
        {{
            "name": "account_access",
            "description": "Customer cannot log in or access their account",
            "keywords": ["can't log in", "locked out", "password reset"]
        }}
    ]
}}"""

    result = generate_json(prompt, temperature=0.3, max_tokens=1024)

    if "parse_error" in result:
        # Fallback: try again with simpler prompt
        prompt2 = f"""List 10 customer support intent categories for {brand_name} as JSON.
Format: {{"intents": [{{"name": "...", "description": "...", "keywords": ["..."]}}]}}"""
        result = generate_json(prompt2, temperature=0.2, max_tokens=1024)

    return result


def save_taxonomy(taxonomy: dict, path: str = None):
    """Save intent taxonomy to disk."""
    if path is None:
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                            "data", "intent_taxonomy.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(taxonomy, f, indent=2)
    print(f"Saved taxonomy to {path}")


class TrivialClassifier:
    """Majority class baseline classifier."""

    def __init__(self, mode: str = "majority"):
        self.mode = mode
        self.majority_class = None
        self.classes = []

    def fit(self, messages: list[str], labels: list[str]):
        """
        `labels` must be the observed label sequence (one entry per training
        example), NOT a deduplicated list of class names. Fitting on a
        deduplicated list makes every class tie at count 1, so most_common()
        silently returns whichever class happens to be first, and the "majority
        class" baseline stops being a majority-class baseline.
        """
        counter = Counter(labels)
        if not counter:
            raise ValueError("TrivialClassifier.fit received no labels")
        self.majority_class = counter.most_common(1)[0][0]
        self.classes = list(counter.keys())
        self.label_counts = dict(counter)

    def predict(self, message: str) -> dict:
        if self.mode == "majority":
            return {"intent": self.majority_class, "confidence": 1.0 / len(self.classes)}
        else:
            return {"intent": random.choice(self.classes), "confidence": 1.0 / len(self.classes)}

    def predict_batch(self, messages: list[str]) -> list[dict]:
        return [self.predict(m) for m in messages]


class TfidfClassifier:
    """Baseline 2: TF-IDF + Logistic Regression."""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words="english",
                                          ngram_range=(1, 2))
        self.model = LogisticRegression(max_iter=1000, random_state=42)
        self.fitted = False

    def fit(self, messages: list[str], labels: list[str]):
        X = self.vectorizer.fit_transform(messages)
        self.model.fit(X, labels)
        self.fitted = True

    def predict(self, message: str) -> dict:
        if not self.fitted:
            return {"intent": "unknown", "confidence": 0.0}
        X = self.vectorizer.transform([message])
        pred = self.model.predict(X)[0]
        proba = self.model.predict_proba(X).max()
        return {"intent": pred, "confidence": round(float(proba), 3)}

    def predict_batch(self, messages: list[str]) -> list[dict]:
        if not self.fitted:
            return [{"intent": "unknown", "confidence": 0.0} for _ in messages]
        X = self.vectorizer.transform(messages)
        preds = self.model.predict(X)
        probas = self.model.predict_proba(X).max(axis=1)
        return [{"intent": p, "confidence": round(float(c), 3)}
                for p, c in zip(preds, probas)]


class LLMClassifier:
    """Main system: LLM few-shot intent classification."""

    # The taxonomy's designated catch-all, used when the LLM returns something
    # unparseable or off-taxonomy.
    FALLBACK_INTENT = "general_inquiry_features"

    def __init__(self, taxonomy: dict = None, model: str = "llama3.2", use_groq: bool = False):
        self.taxonomy = taxonomy or load_taxonomy()
        self.model = model
        self.use_groq = use_groq
        names = [i["name"] for i in self.taxonomy.get("intents", [])]
        self.fallback_intent = (self.FALLBACK_INTENT if self.FALLBACK_INTENT in names
                                else (names[0] if names else "unknown"))

    def _build_prompt(self, message: str) -> str:
        intents = self.taxonomy.get("intents", [])
        intent_descriptions = "\n".join(
            f"- {intent['name']}: {intent['description']}"
            for intent in intents
        )

        return f"""Classify the customer message into one intent category:
{intent_descriptions}

Message: "{message[:250]}"

Respond with JSON: {{"intent": "<intent_name>", "confidence": <0.0-1.0>}}"""

    def predict(self, message: str) -> dict:
        prompt = self._build_prompt(message)
        result = generate_json(prompt, model=self.model, temperature=0.1,
                               max_tokens=80, use_groq=self.use_groq)

        valid_intents = [i["name"] for i in self.taxonomy.get("intents", [])]

        if result.get("failed"):
            return {
                "intent": "unknown",
                "confidence": 0.0,
                "reasoning": result.get("error", "API request failed"),
                "failed": True,
            }

        if "parse_error" in result:
            return {"intent": self.fallback_intent, "confidence": 0.0,
                    "reasoning": "Parse error", "fallback": True}

        if result.get("intent") not in valid_intents:
            predicted = result.get("intent", "").lower().replace(" ", "_")
            for vi in valid_intents:
                if predicted and (predicted in vi or vi in predicted):
                    result["intent"] = vi
                    break
            else:
                result["intent"] = self.fallback_intent
                result["confidence"] = 0.0
                result["fallback"] = True

        return result

    def predict_batch(self, messages: list[str]) -> list[dict]:
        """Predict intents for a batch of messages (sequential for Ollama)."""
        results = []
        for msg in messages:
            results.append(self.predict(msg))
        return results


def create_training_data_from_threads(threads: list[dict], taxonomy: dict,
                                       n_samples: int = 300) -> tuple:
    """
    Use LLM to label a sample of threads for TF-IDF baseline training.
    Returns (messages, labels) tuple.
    """
    sample = random.sample(threads, min(n_samples, len(threads)))
    classifier = LLMClassifier(taxonomy)

    messages = []
    labels = []

    for thread in sample:
        msg = thread["customer_message"]
        result = classifier.predict(msg)
        if result["intent"] != "unknown":
            messages.append(msg)
            labels.append(result["intent"])

    return messages, labels
