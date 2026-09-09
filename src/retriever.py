"""
TF-IDF based retriever for finding similar historical conversations.
Used to ground the reply generator in actual brand behavior.
"""

import os
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ConversationRetriever:
    """
    Retrieves similar historical conversations using TF-IDF cosine similarity.
    No external embedding API needed — runs entirely locally.
    """

    def __init__(self, threads: list[dict] = None):
        self.threads = threads or []
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        self.tfidf_matrix = None
        self._fitted = False

    def fit(self, threads: list[dict] = None):
        """Build TF-IDF index from conversation threads."""
        if threads:
            self.threads = threads

        if not self.threads:
            raise ValueError("No threads to index")

        messages = [t["customer_message"] for t in self.threads]
        self.tfidf_matrix = self.vectorizer.fit_transform(messages)
        self._fitted = True
        print(f"Indexed {len(messages)} conversations "
              f"(vocab size: {len(self.vectorizer.vocabulary_)})")

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Find top-k most similar conversations to the query.
        
        Returns list of dicts with:
        - customer_message: the similar customer query
        - brand_reply: how the brand actually responded
        - similarity: cosine similarity score
        - thread: full thread data
        """
        if not self._fitted:
            raise RuntimeError("Retriever not fitted. Call fit() first.")

        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix)[0]

        top_indices = np.argsort(similarities)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            sim = float(similarities[idx])
            if sim < 0.01:
                continue
            thread = self.threads[idx]
            results.append({
                "customer_message": thread["customer_message"],
                "brand_reply": thread["brand_reply"],
                "similarity": round(sim, 4),
                "num_turns": thread.get("num_turns", 2),
            })

        return results

    def save(self, path: str):
        """Save fitted retriever state."""
        import pickle
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "vectorizer": self.vectorizer,
                "tfidf_matrix": self.tfidf_matrix,
                "threads": self.threads,
            }, f)

    def load(self, path: str):
        """Load fitted retriever state."""
        import pickle
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.vectorizer = state["vectorizer"]
        self.tfidf_matrix = state["tfidf_matrix"]
        self.threads = state["threads"]
        self._fitted = True
