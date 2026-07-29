from json import JSONDecoder
from pathlib import Path
from collections import Counter
import subprocess, sys, json

sys.path.insert(0, "scripts")
from importlib.util import spec_from_file_location, module_from_spec
spec = spec_from_file_location("s01", "scripts/01_scrape_apify.py")
mod = module_from_spec(spec)
spec.loader.exec_module(mod)
apify = mod.resolve_apify_cmd()

raw = Path("data/raw/apify_runs_home.json").read_text(encoding="utf-8-sig", errors="replace")
i = raw.find("[")
runs, _ = JSONDecoder().raw_decode(raw[i:])
print("recent_runs", len(runs))
print("by_status", dict(Counter(r.get("status") for r in runs)))

running = [r for r in runs if r.get("status") in ("RUNNING", "READY")]
print("running_count", len(running))
for r in runs[:15]:
    ds = r.get("defaultDatasetId")
    n = "?"
    if ds:
        p = subprocess.run(
            apify + ["datasets", "info", ds, "--json"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        b = (p.stdout or "") + "\n" + (p.stderr or "")
        j = b.find("{")
        if j >= 0:
            try:
                info, _ = JSONDecoder().raw_decode(b[j:])
                n = info.get("itemCount")
            except Exception:
                pass
    msg = str(r.get("statusMessage") or "")[:70]
    print(f"  {r.get('status'):10} n={str(n):5} {r.get('id')} {msg}")

reg = Path("data/raw/seen_post_ids.json")
print("registry", json.loads(reg.read_text()).get("count") if reg.exists() else 0)
