import json
from pathlib import Path
import pandas as pd

summary = json.loads(Path("data/processed/phase1_summary.json").read_text(encoding="utf-8"))
print(json.dumps(summary, indent=2, default=str)[:2500])

for c in ("a", "b"):
    df = pd.read_parquet(f"data/processed/corpus_{c}.parquet")
    print(f"\n=== CORPUS {c.upper()} ===")
    print("rows", len(df))
    print("cols", list(df.columns))
    print("subreddits", df["subreddit"].value_counts().to_dict())
    print("word_count mean/median", float(df["word_count"].mean()), float(df["word_count"].median()))
    print("word_count_clean mean", float(df["word_count_clean"].mean()))
    print("empty_clean", int((df["text_clean"].fillna("") == "").sum()))
    print("min_words", int(df["word_count"].min()), "max", int(df["word_count"].max()))
    print("date_min", df["created_at"].min(), "date_max", df["created_at"].max())
    # sample
    row = df.sort_values("word_count", ascending=False).iloc[0]
    print("longest title:", str(row["title"])[:80])
    print("text_clean sample:", str(row["text_clean"])[:200])

# interim filter summary
fs = Path("data/interim/filter_summary.json")
if fs.exists():
    print("\nfilter_summary", fs.read_text(encoding="utf-8")[:1500])
