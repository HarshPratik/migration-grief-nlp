import json
from pathlib import Path
from collections import Counter
files = list(Path("data/raw/corpus_a").glob("*.json")) + list(Path("data/raw/corpus_b").glob("*.json"))
ids = []
by_c = Counter()
by_sub = Counter()
good_runs = 0
for p in sorted(files, key=lambda x: x.stat().st_mtime):
    d = json.loads(p.read_text(encoding="utf-8"))
    items = d.get("items") or []
    meta = d.get("meta") or {}
    c = meta.get("corpus") or ("a" if "corpus_a" in str(p) else "b")
    n = len(items)
    if n >= 20:
        good_runs += 1
    by_c[c] += n
    for it in items:
        pid = str(it.get("parsedId") or it.get("id") or "").removeprefix("t3_")
        if pid:
            ids.append(pid)
        sub = str(it.get("parsedCommunityName") or it.get("subreddit") or it.get("communityName") or "?").replace("r/", "")
        by_sub[sub] += 1
print("files", len(files))
print("raw_rows", len(ids))
print("unique", len(set(ids)))
print("by_corpus_rows", dict(by_c))
print("by_sub", dict(by_sub))
print("files_with_20plus", good_runs)
reg = Path("data/raw/seen_post_ids.json")
if reg.exists():
    print("registry", json.loads(reg.read_text(encoding="utf-8")).get("count"))
