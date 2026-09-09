"""
Data processor for the Twitter Customer Support dataset.
Handles: CSV loading, thread reconstruction, brand extraction, and subsampling.
"""

import os
import json
import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm import tqdm


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")


def find_csv_path() -> str:
    """Find the dataset CSV file."""
    base = os.path.dirname(os.path.dirname(__file__))
    candidates = [
        os.path.join(base, "twcs.csv"),
        os.path.join(RAW_DIR, "twcs.csv"),
        os.path.join(base, "data", "twcs.csv"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        "Cannot find twcs.csv. Place it in the project root or data/raw/ directory."
    )


def load_raw_data(nrows: int = None) -> pd.DataFrame:
    """Load the raw CSV with proper dtypes."""
    csv_path = find_csv_path()
    print(f"Loading data from {csv_path}...")

    dtype = {
        "tweet_id": str,
        "author_id": str,
        "inbound": str,
        "text": str,
        "response_tweet_id": str,
        "in_response_to_tweet_id": str,
    }

    df = pd.read_csv(csv_path, dtype=dtype, nrows=nrows)
    df["inbound"] = df["inbound"].map({"True": True, "False": False})
    df["text"] = df["text"].fillna("")

    print(f"Loaded {len(df):,} tweets")
    return df


def extract_brand_from_outbound(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify brand handles: outbound tweets (inbound=False) come from brand accounts.
    We map author_id -> brand for outbound tweets, then propagate to threads.
    """
    outbound = df[df["inbound"] == False]
    brand_authors = outbound["author_id"].unique()
    print(f"Found {len(brand_authors)} unique brand author IDs")
    return brand_authors


def get_brand_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-brand statistics for brand selection."""
    outbound = df[df["inbound"] == False]
    inbound = df[df["inbound"] == True]

    brand_outbound_counts = outbound["author_id"].value_counts().reset_index()
    brand_outbound_counts.columns = ["brand_author_id", "outbound_count"]

    stats = []
    for _, row in brand_outbound_counts.head(20).iterrows():
        brand_id = row["brand_author_id"]
        brand_out = outbound[outbound["author_id"] == brand_id]
        n_outbound = len(brand_out)

        responded_to_ids = brand_out["in_response_to_tweet_id"].dropna().unique()
        n_conversations = len(responded_to_ids)
        sample_replies = brand_out["text"].sample(min(5, len(brand_out))).tolist()
        avg_reply_len = brand_out["text"].str.len().mean()

        stats.append({
            "brand_author_id": brand_id,
            "outbound_count": n_outbound,
            "conversations": n_conversations,
            "avg_reply_length": round(avg_reply_len, 1),
            "sample_reply": sample_replies[0] if sample_replies else "",
        })

    return pd.DataFrame(stats).sort_values("outbound_count", ascending=False)


def reconstruct_threads(df: pd.DataFrame, brand_id: str) -> list[dict]:
    """
    Reconstruct multi-turn conversation threads for a specific brand.
    
    Returns a list of conversation dicts:
    {
        "thread_id": str,
        "messages": [
            {"tweet_id": str, "author_id": str, "text": str, "inbound": bool, "created_at": str},
            ...
        ],
        "customer_message": str,  # The initial inbound message
        "brand_reply": str,       # The brand's response
    }
    """
    brand_tweets = df[df["author_id"] == brand_id]
    brand_tweet_ids = set(brand_tweets["tweet_id"].values)

    tweet_lookup = {}
    for _, row in df.iterrows():
        tweet_lookup[str(row["tweet_id"])] = row

    response_map = defaultdict(list)
    for _, row in df.iterrows():
        resp_ids = row.get("response_tweet_id", "")
        if pd.notna(resp_ids) and resp_ids:
            for rid in str(resp_ids).split(","):
                rid = rid.strip()
                if rid:
                    response_map[str(row["tweet_id"])].append(rid)

    threads = []
    seen_tweet_ids = set()

    for _, brand_tweet in brand_tweets.iterrows():
        in_resp_to = brand_tweet.get("in_response_to_tweet_id")
        if pd.isna(in_resp_to):
            continue

        in_resp_to = str(int(float(in_resp_to)))

        if in_resp_to in seen_tweet_ids:
            continue
        seen_tweet_ids.add(in_resp_to)

        customer_tweet = tweet_lookup.get(in_resp_to)
        if customer_tweet is None or customer_tweet.get("inbound") != True:
            continue

        customer_text = str(customer_tweet.get("text", ""))
        brand_text = str(brand_tweet.get("text", ""))

        if not customer_text.strip() or not brand_text.strip():
            continue

        messages = [
            {
                "tweet_id": str(customer_tweet["tweet_id"]),
                "author_id": str(customer_tweet["author_id"]),
                "text": customer_text,
                "inbound": True,
            },
            {
                "tweet_id": str(brand_tweet["tweet_id"]),
                "author_id": str(brand_tweet["author_id"]),
                "text": brand_text,
                "inbound": False,
            },
        ]

        current_id = str(brand_tweet["tweet_id"])
        for _ in range(4):
            responses = response_map.get(current_id, [])
            if not responses:
                break
            next_id = responses[0]
            next_tweet = tweet_lookup.get(next_id)
            if next_tweet is None:
                break
            messages.append({
                "tweet_id": str(next_tweet["tweet_id"]),
                "author_id": str(next_tweet["author_id"]),
                "text": str(next_tweet.get("text", "")),
                "inbound": bool(next_tweet.get("inbound")),
            })
            current_id = next_id

        threads.append({
            "thread_id": in_resp_to,
            "messages": messages,
            "customer_message": customer_text,
            "brand_reply": brand_text,
            "num_turns": len(messages),
        })

    print(f"Reconstructed {len(threads)} conversation threads for brand {brand_id}")
    return threads


def create_brand_subsample(df: pd.DataFrame, brand_id: str,
                           max_threads: int = 5000) -> list[dict]:
    """
    Create a manageable subsample of threads for a brand.
    Prioritizes diversity: varied message lengths, multi-turn threads, etc.
    """
    threads = reconstruct_threads(df, brand_id)

    if len(threads) <= max_threads:
        return threads

    np.random.seed(42)

    threads_sorted = sorted(threads, key=lambda t: len(t["customer_message"]))
    n = len(threads_sorted)

    third = max_threads // 3
    indices = (
        list(range(0, min(third, n // 3))) +
        list(range(n // 3, min(n // 3 + third, 2 * n // 3))) +
        list(range(2 * n // 3, min(2 * n // 3 + third, n)))
    )

    remaining = max_threads - len(indices)
    if remaining > 0:
        available = [i for i in range(n) if i not in set(indices)]
        indices.extend(np.random.choice(available,
                                        size=min(remaining, len(available)),
                                        replace=False).tolist())

    subsample = [threads_sorted[i] for i in indices[:max_threads]]
    print(f"Created subsample of {len(subsample)} threads (from {len(threads)} total)")
    return subsample


def save_processed_data(threads: list[dict], brand_id: str):
    """Save processed threads to disk."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    output_path = os.path.join(PROCESSED_DIR, f"{brand_id}_threads.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(threads, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(threads)} threads to {output_path}")
    return output_path


def load_processed_data(brand_id: str) -> list[dict]:
    """Load previously processed threads."""
    path = os.path.join(PROCESSED_DIR, f"{brand_id}_threads.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No processed data found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    df = load_raw_data()
    stats = get_brand_statistics(df)
    print("\n=== Top 15 Brands by Outbound Volume ===")
    print(stats.head(15).to_string(index=False))
    print(f"\nTotal tweets: {len(df):,}")
    print(f"Inbound: {(df['inbound'] == True).sum():,}")
    print(f"Outbound: {(df['inbound'] == False).sum():,}")
