from json import JSONDecoder
from pathlib import Path
from collections import Counter

raw = Path("data/raw/apify_runs_memcheck.json").read_text(encoding="utf-8-sig", errors="replace")
i = raw.find("[")
runs, _ = JSONDecoder().raw_decode(raw[i:])
print("by_status", dict(Counter(r.get("status") for r in runs)))
active = [r for r in runs if r.get("status") in ("RUNNING", "READY")]
print("active", len(active))
mem_total = 0
for r in active:
    opts = r.get("options") or {}
    m = opts.get("memoryMbytes") or 2048
    mem_total += m
    print(f"  {r.get('status')} {r.get('id')} mem={m}MB")
print(f"active_memory_MB={mem_total}  headroom_to_16GB={16384 - mem_total}")
print(f"theoretical_max_concurrent_at_2GB={16384 // 2048}")
print(f"theoretical_max_at_1GB={16384 // 1024}")
