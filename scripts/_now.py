from pathlib import Path
import json
from collections import Counter
from datetime import datetime

reg = Path("data/raw/seen_post_ids.json")
print("registry", json.loads(reg.read_text()).get("count") if reg.exists() else 0)

ids=set(); raw=0; by_sub=Counter()
for p in Path("data/raw").rglob("*.json"):
    if p.name.startswith(("scrape_log","apify_","seen")) or "manifest" in p.name: continue
    if any(p.name.startswith(x) for x in ("full_","home_","fire_","fresh_","cloud_d")): continue
    try: d=json.loads(p.read_text(encoding="utf-8"))
    except: continue
    items=d.get("items") if isinstance(d,dict) else d
    if not isinstance(items,list): continue
    for it in items:
        if not isinstance(it,dict): continue
        pid=str(it.get("parsedId") or it.get("id") or "").removeprefix("t3_")
        if not pid: continue
        ids.add(pid); raw+=1
        sub=str(it.get("parsedCommunityName") or it.get("subreddit") or it.get("communityName") or "?").replace("r/","")
        by_sub[sub]+=1
print("unique_on_disk", len(ids), "raw_rows", raw)
print("by_sub", dict(by_sub.most_common()))

for label in ("interim","processed"):
    folder=Path("data")/label
    for c in ("a","b"):
        pq=folder/f"corpus_{c}.parquet"
        if pq.exists():
            import pandas as pd
            print(label, c, len(pd.read_parquet(pq)))

logs=sorted(Path("data/raw").glob("scrape_log_full_*.json"), key=lambda p:p.stat().st_mtime)
if logs:
    s=json.loads(logs[-1].read_text())
    print("last_session", logs[-1].name, "raw", s.get("total_raw"), "new", s.get("total_new"), "registry", s.get("registry_size"), "budget", s.get("budget_exhausted"))
