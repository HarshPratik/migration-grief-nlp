from pathlib import Path
import json
from collections import Counter, defaultdict
from datetime import datetime

ids = set()
by_corpus = Counter()
by_sub = Counter()
raw_rows = 0
files = 0
file_items = []

skip_names = {
    "seen_post_ids.json",
}
skip_prefixes = ("scrape_log", "apify_", "fire_and", "full_scrape", "home_batch", "fresh_batch", "cloud_download")
skip_suffixes = ("_report.json", "_list.json", "_memcheck.json", "_home.json", "_status_report.json")

for p in Path("data/raw").rglob("*.json"):
    name = p.name
    if name in skip_names or name.startswith(".") or any(name.startswith(x) for x in skip_prefixes):
        continue
    if any(name.endswith(x) for x in skip_suffixes):
        continue
    if "manifest" in name:
        continue
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        continue
    items = d.get("items") if isinstance(d, dict) else d
    if not isinstance(items, list) or not items:
        continue
    meta = d.get("meta") if isinstance(d, dict) else {}
    files += 1
    n = 0
    for it in items:
        if not isinstance(it, dict):
            continue
        pid = str(it.get("parsedId") or it.get("id") or "").removeprefix("t3_")
        if not pid:
            continue
        n += 1
        raw_rows += 1
        ids.add(pid)
        sub = (
            it.get("parsedCommunityName")
            or it.get("subreddit")
            or it.get("communityName")
            or meta.get("subreddit")
            or "?"
        )
        sub = str(sub).replace("r/", "").strip() or "?"
        by_sub[sub] += 1
        # corpus guess
        if sub in ("expats", "immigration", "StudyAbroad"):
            by_corpus["A"] += 1
        elif sub in ("GriefSupport", "bereavement"):
            by_corpus["B"] += 1
        else:
            # path hint
            if "corpus_a" in str(p):
                by_corpus["A"] += 1
            elif "corpus_b" in str(p):
                by_corpus["B"] += 1
            else:
                by_corpus["?"] += 1
    file_items.append((n, p))

reg = Path("data/raw/seen_post_ids.json")
reg_n = json.loads(reg.read_text()).get("count") if reg.exists() else None

# interim / processed
interim = {}
processed = {}
for label, folder in [("interim", Path("data/interim")), ("processed", Path("data/processed"))]:
    for c in ("a", "b"):
        pq = folder / f"corpus_{c}.parquet"
        csv = folder / f"corpus_{c}.csv"
        if pq.exists():
            try:
                import pandas as pd
                df = pd.read_parquet(pq)
                (interim if label == "interim" else processed)[c] = len(df)
            except Exception as e:
                (interim if label == "interim" else processed)[c] = f"err:{e}"
        elif csv.exists():
            try:
                import pandas as pd
                df = pd.read_csv(csv)
                (interim if label == "interim" else processed)[c] = len(df)
            except Exception as e:
                (interim if label == "interim" else processed)[c] = f"err:{e}"

print("=== RAW SCRAPE DATA ===")
print("dump_files_with_items", files)
print("raw_rows_total", raw_rows)
print("unique_post_ids", len(ids))
print("registry_count", reg_n)
print("by_corpus_rows", dict(by_corpus))
print("by_subreddit_rows", dict(by_sub.most_common()))
print("top_dumps_by_size", [(n, str(p.relative_to("data/raw"))) for n, p in sorted(file_items, reverse=True)[:8]])
print("interim_filtered", interim)
print("processed_nlp", processed)

# folder sizes
for sub in ["corpus_a", "corpus_b", "cloud", "pilot"]:
    d = Path("data/raw") / sub
    if d.exists():
        sz = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
        print(f"size_{sub}_MB", round(sz / 1e6, 2))
