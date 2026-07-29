"""Download all datasets from apify_runs_status_report.json into data/raw/cloud/."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from json import JSONDecoder
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from seen_ids import SeenIdRegistry  # noqa: E402
from importlib.util import module_from_spec, spec_from_file_location

spec = spec_from_file_location("s01", ROOT / "scripts" / "01_scrape_apify.py")
mod = module_from_spec(spec)
spec.loader.exec_module(mod)
apify = mod.resolve_apify_cmd()

REPORT = ROOT / "data" / "raw" / "apify_runs_status_report.json"
OUT = ROOT / "data" / "raw" / "cloud"


def fetch_items(dataset_id: str) -> list:
    proc = subprocess.run(
        apify + ["datasets", "get-items", dataset_id, "--format", "json"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    blob = (proc.stdout or "") + "\n" + (proc.stderr or "")
    j = blob.find("[")
    if j < 0:
        return []
    items, _ = JSONDecoder().raw_decode(blob[j:])
    return items if isinstance(items, list) else []


def main() -> int:
    rows = json.loads(REPORT.read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    registry = SeenIdRegistry(seed_from_dumps=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    total_raw = total_new = total_dup = 0
    for row in rows:
        ds = row.get("dataset_id")
        n = row.get("n_items") or 0
        if not ds or not isinstance(n, int) or n <= 0:
            print(f"SKIP {row.get('id')} n={n}")
            continue
        out = OUT / f"{row['status']}_{row['id']}_{ds}.json"
        if out.exists() and out.stat().st_size > 100:
            print(f"HAVE {out.name}")
            # still count into registry if needed
            try:
                data = json.loads(out.read_text(encoding="utf-8"))
                items = data.get("items") or []
            except Exception:
                items = fetch_items(ds)
        else:
            print(f"GET  {ds} (expect ~{n}) …", flush=True)
            items = fetch_items(ds)
            meta = {
                "apify_run_id": row.get("id"),
                "dataset_id": ds,
                "status": row.get("status"),
                "n_items_reported": n,
                "scraped_at": stamp,
                "source": "cloud_bulk_download",
            }
            out.write_text(
                json.dumps({"meta": meta, "items": items}, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
        new_items, n_new, n_dup = registry.filter_new(items)
        total_raw += len(items)
        total_new += n_new
        total_dup += n_dup
        print(f"  raw={len(items)} new={n_new} dup={n_dup}", flush=True)
        registry.save()
    print(
        f"\nDone raw={total_raw} new={total_new} dup={total_dup} registry={len(registry)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
