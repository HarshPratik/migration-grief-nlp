from pathlib import Path
import json
from collections import Counter
from json import JSONDecoder
import subprocess, sys

# parse log-like summary from scrape logs
logs = sorted(Path("data/raw").glob("scrape_log_full_*.json"), key=lambda p: p.stat().st_mtime)
if logs:
    s = json.loads(logs[-1].read_text(encoding="utf-8"))
    print("latest_log", logs[-1].name)
    for k in ["workers", "total_raw", "total_new", "total_dup", "registry_size", "session_seconds", "budget_exhausted"]:
        print(k, s.get(k))
    for r in s.get("runs") or []:
        st = r.get("status")
        if st in ("ok", "failed", "empty", "budget_exhausted", "concurrency_limit"):
            print(st, r.get("run_name"), "raw", r.get("n_raw"), "new", r.get("n_new"), "dup", r.get("n_dup"), "s", r.get("seconds"), (r.get("error") or "")[:100])

reg = Path("data/raw/seen_post_ids.json")
print("registry", json.loads(reg.read_text()).get("count") if reg.exists() else 0)

# process check
