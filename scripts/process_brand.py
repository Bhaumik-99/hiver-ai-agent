"""
Fast data processing for AppleSupport.
Uses vectorized pandas operations instead of row-by-row iteration.
"""
import sys
import os
import json
import random
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np

BRAND_ID = "AppleSupport"
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

def main():
    start = time.time()
    
    csv_path = os.path.join(BASE_DIR, "twcs.csv")
    print(f"Loading {csv_path}...")
    
    dtype = {
        "tweet_id": str,
        "author_id": str,
        "inbound": str,
        "text": str,
        "response_tweet_id": str,
        "in_response_to_tweet_id": str,
    }
    df = pd.read_csv(csv_path, dtype=dtype)
    df["inbound"] = df["inbound"].map({"True": True, "False": False})
    df["text"] = df["text"].fillna("")
    print(f"Loaded {len(df):,} tweets in {time.time()-start:.1f}s")
    
    t0 = time.time()
    
    brand_mask = (df["author_id"] == BRAND_ID) & (df["inbound"] == False)
    brand_tweets = df[brand_mask].copy()
    print(f"AppleSupport outbound tweets: {len(brand_tweets):,}")
    
    brand_tweets["in_response_to_tweet_id"] = brand_tweets["in_response_to_tweet_id"].astype(str)
    brand_tweets["resp_to_clean"] = brand_tweets["in_response_to_tweet_id"].apply(
        lambda x: str(int(float(x))) if x != "nan" and x != "" else ""
    )
    brand_tweets = brand_tweets[brand_tweets["resp_to_clean"] != ""]
    
    customer_tweet_ids = set(brand_tweets["resp_to_clean"].values)
    print(f"Customer tweets Apple responded to: {len(customer_tweet_ids):,}")
    
    df["tweet_id_str"] = df["tweet_id"].astype(str)
    customer_tweets = df[df["tweet_id_str"].isin(customer_tweet_ids) & (df["inbound"] == True)].copy()
    print(f"Found matching customer tweets: {len(customer_tweets):,}")
    
    customer_lookup = dict(zip(customer_tweets["tweet_id_str"], customer_tweets["text"]))
    
    threads = []
    seen = set()
    
    for _, row in brand_tweets.iterrows():
        cust_id = row["resp_to_clean"]
        if cust_id in seen:
            continue
        seen.add(cust_id)
        
        cust_text = customer_lookup.get(cust_id, "")
        brand_text = row["text"]
        
        if not cust_text.strip() or not brand_text.strip():
            continue
        
        threads.append({
            "thread_id": cust_id,
            "customer_message": str(cust_text),
            "brand_reply": str(brand_text),
            "num_turns": 2,
            "messages": [
                {"text": str(cust_text), "inbound": True, "tweet_id": cust_id},
                {"text": str(brand_text), "inbound": False, "tweet_id": str(row["tweet_id"])},
            ]
        })
    
    print(f"Built {len(threads):,} conversation threads in {time.time()-t0:.1f}s")
    
    random.seed(42)
    MAX_THREADS = 5000
    
    if len(threads) > MAX_THREADS:
        subsample = random.sample(threads, MAX_THREADS)
    else:
        subsample = threads
    
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    output_path = os.path.join(PROCESSED_DIR, f"{BRAND_ID}_threads.json")
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(subsample, f, indent=2, ensure_ascii=False)
    
    print(f"\nSaved {len(subsample)} threads to {output_path}")
    
    msg_lens = [len(t["customer_message"]) for t in subsample]
    reply_lens = [len(t["brand_reply"]) for t in subsample]
    
    print(f"\n=== Subsample Statistics ===")
    print(f"Threads: {len(subsample)}")
    print(f"Customer msg length: mean={np.mean(msg_lens):.0f}, "
          f"min={min(msg_lens)}, max={max(msg_lens)}")
    print(f"Brand reply length: mean={np.mean(reply_lens):.0f}, "
          f"min={min(reply_lens)}, max={max(reply_lens)}")
    
    print("\n=== Sample Conversations ===")
    for i, t in enumerate(subsample[:5]):
        cust = t["customer_message"][:150].encode("ascii", "replace").decode()
        reply = t["brand_reply"][:150].encode("ascii", "replace").decode()
        print(f"\n  {i+1}. Customer: {cust}")
        print(f"     Apple:    {reply}")
    
    print(f"\nTotal time: {time.time()-start:.1f}s")


if __name__ == "__main__":
    main()
