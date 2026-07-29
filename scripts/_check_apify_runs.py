import json, re, subprocess, sys
from pathlib import Path
from collections import Counter

raw = Path("data/raw/apify_runs_list.json").read_text(encoding="utf-8-sig", errors="replace")
# strip powershell noise
i = raw.find("[")
if i < 0:
    i = raw.find("{")
text = raw[i:] if i >= 0 else raw
try:
    data = json.loads(text)
except json.JSONDecodeError:
    # try find array
    m = re.search(r"\[.*\]", text, re.S)
    data = json.loads(m.group(0)) if m else []

if isinstance(data, dict):
    runs = data.get("data") or data.get("items") or data.get("runs") or [data]
else:
    runs = data

print(f"Listed runs: {len(runs)}")
status_c = Counter()
for r in runs:
    status_c[r.get("status") or r.get("statusMessage") or "?"] += 1
print("By status:", dict(status_c))

# resolve apify cmd
sys.path.insert(0, "scripts")
from importlib.util import spec_from_file_location, module_from_spec
spec = spec_from_file_location("s01", "scripts/01_scrape_apify.py")
mod = module_from_spec(spec)
spec.loader.exec_module(mod)
apify = mod.resolve_apify_cmd()

results = []
for r in runs:
    rid = r.get("id")
    status = r.get("status")
    ds = r.get("defaultDatasetId")
    started = r.get("startedAt") or r.get("started_at")
    finished = r.get("finishedAt") or r.get("finished_at")
    usage = r.get("usageTotalUsd") or (r.get("usage") or {}).get("TOTAL") if isinstance(r.get("usage"), dict) else r.get("usageTotalUsd")
    n_items = None
    if ds and status in ("SUCCEEDED", "TIMED-OUT", "ABORTED", "FAILED") or (ds and status):
        # get item count via dataset info if possible
        proc = subprocess.run(
            apify + ["datasets", "get-items", ds, "--format", "json"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        blob = (proc.stdout or "") + "\n" + (proc.stderr or "")
        # count array length without full parse if huge
        j = blob.find("[")
        if j >= 0:
            try:
                # strip content-type noise
                items = json.loads(blob[j:]) if blob[j:].strip().startswith("[") else None
                if items is None:
                    # try JSONDecoder
                    from json import JSONDecoder
                    items, _ = JSONDecoder().raw_decode(blob[j:])
                n_items = len(items) if isinstance(items, list) else None
            except Exception as e:
                n_items = f"err:{e}"
    results.append({
        "id": rid,
        "status": status,
        "dataset_id": ds,
        "n_items": n_items,
        "startedAt": started,
        "finishedAt": finished,
        "usageTotalUsd": r.get("usageTotalUsd"),
    })
    print(f"{status:12} items={str(n_items):6} run={rid} ds={ds} started={started}")

Path("data/raw/apify_runs_status_report.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
terminal = [x for x in results if x["status"] in ("SUCCEEDED", "TIMED-OUT", "FAILED", "ABORTED")]
running = [x for x in results if x["status"] in ("RUNNING", "READY", "PENDING")]
total_items = sum(x["n_items"] for x in results if isinstance(x["n_items"], int))
print("\n=== SUMMARY ===")
print("terminal", len(terminal), "running/ready", len(running), "sum_items", total_items)
