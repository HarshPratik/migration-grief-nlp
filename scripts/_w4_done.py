from pathlib import Path
import json
from collections import Counter
from datetime import datetime

logs = sorted(Path("data/raw").glob("scrape_log_full_*.json"), key=lambda p: p.stat().st_mtime)
if logs:
    s = json.loads(logs[-1].read_text(encoding="utf-8"))
    print("log", logs[-1].name)
    for k in ["workers", "total_raw", "total_new", "total_dup", "registry_size", "session_seconds", "budget_exhausted"]:
        print(k, s.get(k))
    ok = [r for r in s.get("runs") or [] if r.get("status") == "ok"]
    print("ok_runs", len(ok))
    for r in ok:
        print(" ", r.get("run_name"), "raw", r.get("n_raw"), "new", r.get("n_new"), "dup", r.get("n_dup"))

reg = json.loads(Path("data/raw/seen_post_ids.json").read_text())
print("registry", reg.get("count"))

# quick unique recount
ids=set(); raw=0
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
        if pid:
            ids.add(pid); raw+=1
print("unique_on_disk", len(ids), "raw_rows", raw)
