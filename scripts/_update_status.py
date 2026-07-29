from pathlib import Path
import json
from collections import Counter
from datetime import datetime

# latest scrape log
logs = sorted(Path("data/raw").glob("scrape_log_full_*.json"), key=lambda p: p.stat().st_mtime)
if logs:
    s = json.loads(logs[-1].read_text(encoding="utf-8"))
    print("latest_log", logs[-1].name, "mtime", datetime.fromtimestamp(logs[-1].stat().st_mtime))
    for k in ["workers", "total_raw", "total_new", "total_dup", "registry_size", "session_seconds", "budget_exhausted"]:
        print(k, s.get(k))
    ok = [r for r in (s.get("runs") or []) if r.get("status") == "ok"]
    failed = [r for r in (s.get("runs") or []) if r.get("status") in ("failed", "empty", "budget_exhausted")]
    skipped = [r for r in (s.get("runs") or []) if r.get("status") == "skipped_resume"]
    print("ok_runs", len(ok), "failed", len(failed), "skipped", len(skipped))
    for r in ok[-10:]:
        print(" OK", r.get("run_name"), "raw", r.get("n_raw"), "new", r.get("n_new"), "dup", r.get("n_dup"), "s", r.get("seconds"))
    for r in failed[-5:]:
        print(" FAIL", r.get("status"), r.get("run_name"), (r.get("error") or "")[:120])

reg = Path("data/raw/seen_post_ids.json")
print("registry", json.loads(reg.read_text()).get("count") if reg.exists() else 0)

# count local dumps
ca = list(Path("data/raw/corpus_a").glob("*.json")) if Path("data/raw/corpus_a").exists() else []
cb = list(Path("data/raw/corpus_b").glob("*.json")) if Path("data/raw/corpus_b").exists() else []
cloud = list(Path("data/raw/cloud").glob("*.json")) if Path("data/raw/cloud").exists() else []
print("files corpus_a", len(ca), "corpus_b", len(cb), "cloud", len(cloud))
