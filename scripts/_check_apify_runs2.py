import json, subprocess, sys
from pathlib import Path
from json import JSONDecoder
from collections import Counter

sys.path.insert(0, "scripts")
from importlib.util import spec_from_file_location, module_from_spec
spec = spec_from_file_location("s01", "scripts/01_scrape_apify.py")
mod = module_from_spec(spec)
spec.loader.exec_module(mod)
apify = mod.resolve_apify_cmd()

raw = Path("data/raw/apify_runs_list.json").read_text(encoding="utf-8-sig", errors="replace")
i = raw.find("[")
runs, _ = JSONDecoder().raw_decode(raw[i:])
print("runs", len(runs))

def item_count(ds: str) -> int:
    proc = subprocess.run(
        apify + ["datasets", "info", ds, "--json"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    blob = (proc.stdout or "") + "\n" + (proc.stderr or "")
    j = blob.find("{")
    if j < 0:
        return -1
    try:
        info, _ = JSONDecoder().raw_decode(blob[j:])
        return int(info.get("itemCount") or info.get("cleanItemCount") or -1)
    except Exception:
        return -1

rows = []
for r in runs:
    ds = r.get("defaultDatasetId")
    n = item_count(ds) if ds else -1
    rows.append({
        "id": r.get("id"),
        "status": r.get("status"),
        "dataset_id": ds,
        "n_items": n,
        "startedAt": r.get("startedAt"),
        "finishedAt": r.get("finishedAt"),
        "usageTotalUsd": r.get("usageTotalUsd"),
    })
    print(f"{r.get('status'):10} n={n:5} run={r.get('id')}")

Path("data/raw/apify_runs_status_report.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
print("status", dict(Counter(x["status"] for x in rows)))
vals = [x["n_items"] for x in rows if isinstance(x["n_items"], int) and x["n_items"] >= 0]
print("total_items_sum", sum(vals))
print("runs_with_data", sum(1 for v in vals if v > 0))
print("zero_item_runs", sum(1 for v in vals if v == 0))
print("running", sum(1 for x in rows if x["status"] in ("RUNNING", "READY", "PENDING")))
