"""
AI Support Agent orchestrator.
Ties together: intent classification, retrieval, reply generation, and escalation.
"""

import json
import os
import time

from src.intent_classifier import LLMClassifier, TfidfClassifier, TrivialClassifier, load_taxonomy
from src.retriever import ConversationRetriever
from src.reply_generator import LLMReplyGenerator, NearestNeighborReplyGenerator, TrivialReplyGenerator
from src.escalation import decide_escalation, TrivialEscalation, RuleOnlyEscalation


class SupportAgent:
    """
    Full AI support agent pipeline.
    
    Handles an incoming customer message by:
    1. Classifying intent
    2. Retrieving similar historical conversations
    3. Generating a grounded reply
    4. Deciding whether to escalate
    """

    def __init__(self, threads: list[dict], taxonomy: dict,
                 brand_name: str = "AppleSupport", model: str = "llama3.2",
                 use_groq: bool = False):
        self.brand_name = brand_name
        self.model = model
        self.use_groq = use_groq

        self.retriever = ConversationRetriever(threads)
        self.retriever.fit()
        self.classifier = LLMClassifier(taxonomy, model=model, use_groq=use_groq)
        self.generator = LLMReplyGenerator(brand_name=brand_name, model=model, use_groq=use_groq)

    def handle_message(self, customer_message: str,
                       conversation_history: list = None) -> dict:
        start_time = time.time()

        intent_result = self.classifier.predict(customer_message)
        similar = self.retriever.retrieve(customer_message, top_k=5)
        reply = self.generator.generate(
            customer_message,
            intent=intent_result.get("intent"),
            similar_convos=similar,
        )
        escalation = decide_escalation(
            customer_message,
            intent=intent_result.get("intent"),
            conversation_history=conversation_history,
            use_llm=True,
            use_groq=self.use_groq,
        )

        is_failed = bool(intent_result.get("failed") or escalation.get("failed"))
        error_msg = intent_result.get("reasoning") if intent_result.get("failed") else escalation.get("reason")

        res = {
            "customer_message": customer_message,
            "intent": intent_result,
            "reply": reply,
            "escalation": escalation,
            "similar_conversations": similar[:3],
            "processing_time": round(time.time() - start_time, 2),
        }
        if is_failed:
            res["failed"] = True
            res["error"] = error_msg
        return res


class TrivialAgent:
    """Baseline 1: majority intent, template reply, never escalate."""

    def __init__(self, taxonomy: dict, train_labels: list[str] = None,
                 intent_override: dict = None):
        """
        train_labels: the observed intent labels of the TRAINING split. Required —
        the previous version fitted on `["dummy"] * len(intents)` paired with the
        deduplicated taxonomy names, which tied every class at count 1 and made
        "majority class" collapse to whatever sat first in the taxonomy file.
        That understated the floor and inflated every "improvement over trivial".

        intent_override: message -> intent, used to inject out-of-fold predictions
        so this baseline is never scored on examples it was fitted on.
        """
        if not train_labels:
            raise ValueError("TrivialAgent needs the training label sequence to "
                             "determine a real majority class")
        self.classifier = TrivialClassifier(mode="majority")
        self.classifier.fit(["" for _ in train_labels], train_labels)
        self.generator = TrivialReplyGenerator()
        self.escalation = TrivialEscalation()
        self.intent_override = intent_override or {}

    def handle_message(self, customer_message: str,
                       conversation_history: list = None) -> dict:
        start_time = time.time()
        if customer_message in self.intent_override:
            intent_result = self.intent_override[customer_message]
        else:
            intent_result = self.classifier.predict(customer_message)
        reply = self.generator.generate(customer_message, intent=intent_result["intent"])
        escalation = self.escalation.decide(customer_message)
        return {
            "customer_message": customer_message,
            "intent": intent_result,
            "reply": reply,
            "escalation": escalation,
            "similar_conversations": [],
            "processing_time": round(time.time() - start_time, 4),
        }


class SimpleAgent:
    """Baseline 2: TF-IDF intent, nearest-neighbor reply, rule-based escalation."""

    def __init__(self, threads: list[dict], taxonomy: dict, training_data: tuple = None,
                 intent_override: dict = None):
        """
        threads: the retrieval corpus. It MUST already have the evaluation examples
        removed, or the nearest neighbour of a golden message is that message's own
        thread and this baseline returns the reference reply verbatim.

        intent_override: message -> intent, used to inject out-of-fold predictions
        so this baseline is never scored on examples it was fitted on.
        """
        self.retriever = ConversationRetriever(threads)
        self.retriever.fit()

        self.classifier = TfidfClassifier()
        if training_data:
            messages, labels = training_data
            self.classifier.fit(messages, labels)

        self.generator = NearestNeighborReplyGenerator(self.retriever)
        self.escalation = RuleOnlyEscalation()
        self.intent_override = intent_override or {}

    def handle_message(self, customer_message: str,
                       conversation_history: list = None) -> dict:
        start_time = time.time()
        if customer_message in self.intent_override:
            intent_result = self.intent_override[customer_message]
        else:
            intent_result = self.classifier.predict(customer_message)
        similar = self.retriever.retrieve(customer_message, top_k=3)
        reply = self.generator.generate(customer_message, similar_convos=similar)
        escalation = self.escalation.decide(customer_message)
        return {
            "customer_message": customer_message,
            "intent": intent_result,
            "reply": reply,
            "escalation": escalation,
            "similar_conversations": similar[:3],
            "processing_time": round(time.time() - start_time, 4),
        }
