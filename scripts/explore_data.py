"""Quick EDA script to explore brands and select the best one."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.data_processor import load_raw_data, get_brand_statistics

df = load_raw_data()
stats = get_brand_statistics(df)
print("\n=== Top 15 Brands by Outbound Volume ===")
print(stats.head(15).to_string(index=False))
print(f"\nTotal tweets: {len(df):,}")
inb = (df["inbound"] == True).sum()
outb = (df["inbound"] == False).sum()
print(f"Inbound: {inb:,}")
print(f"Outbound: {outb:,}")

# Show sample replies from top 3 brands
print("\n=== Sample Replies from Top 3 Brands ===")
for _, row in stats.head(3).iterrows():
    brand = row["brand_author_id"]
    print(f"\n--- {brand} ---")
    brand_out = df[(df["author_id"] == brand) & (df["inbound"] == False)]
    samples = brand_out["text"].sample(3, random_state=42).tolist()
    for i, s in enumerate(samples, 1):
        print(f"  {i}. {s[:200]}")
