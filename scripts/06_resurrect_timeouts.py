"""
Resurrect TIMED-OUT Apify runs so they continue with a longer timeout budget.

CLI resurrect continues from the request queue (same dataset). Timeout is
re-counted from resurrect time.

  python scripts/06_resurrect_timeouts.py --dry-run
  python scripts/06_resurrect_timeouts.py --max-items-threshold 50 --limit 2
  python scripts/06_resurrect_timeouts.py --run-id dvV0au8yOK25hAUa1

Note: Setting timeout>original requires API token (APIFY_TOKEN) or Console.
CLI resurrect uses the previous timeoutSecs (often 1200) but the clock restarts.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from json import JSONDecoder
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from importlib.util import module_from_spec, spec_from_file_location

spec = spec_from_file_location("s01", ROOT / "scripts" / "01_scrape_apify.py")
mod = module_from_spec(spec)
spec.loader.exec_module(mod)
apify = mod.resolve_apify_cmd()
resolve_token = mod.resolve_token

ACTOR = "trudax/reddit-scraper-lite"


def parse_json(blob: str):
    blob = blob or ""
    i = blob.find("{")
    j = blob.find("[")
    if j >= 0 and (i < 0 or j < i):
        return JSONDecoder().raw_decode(blob[j:])[0]
    if i >= 0:
        return JSONDecoder().raw_decode(blob[i:])[0]
    raise ValueError("no json")


def list_runs(limit: int = 40) -> list:
    proc = subprocess.run(
        apify + ["runs", "ls", ACTOR, "--limit", str(limit), "--desc", "--json"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return parse_json((proc.stdout or "") + "\n" + (proc.stderr or ""))


def item_count(dataset_id: str) -> int:
    if not dataset_id:
        return -1
    proc = subprocess.run(
        apify + ["datasets", "info", dataset_id, "--json"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    try:
        info = parse_json((proc.stdout or "") + "\n" + (proc.stderr or ""))
        return int(info.get("itemCount") or info.get("cleanItemCount") or -1)
    except Exception:
        return -1


def resurrect_cli(run_id: str) -> dict:
    proc = subprocess.run(
        apify + ["runs", "resurrect", run_id, "--json"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    blob = (proc.stdout or "") + "\n" + (proc.stderr or "")
    if "memory limit" in blob.lower():
        raise RuntimeError(f"APIFY_MEMORY_LIMIT: {blob[:500]}")
    try:
        return parse_json(blob)
    except Exception as exc:
        raise RuntimeError(f"resurrect failed: {blob[:800]}") from exc


def resurrect_api(run_id: str, timeout_secs: int, memory_mbytes: int) -> dict:
    """Resurrect with explicit longer timeout via REST API."""
    token = resolve_token()
    if not token:
        # try keyring
        try:
            import keyring

            token = keyring.get_password("com.apify.cli", "token")
        except Exception:
            token = None
    if not token:
        raise RuntimeError(
            "APIFY_TOKEN not set — CLI resurrect used instead (same timeout as original). "
            "Set APIFY_TOKEN for timeout=3600+ on resurrect."
        )
    import urllib.request

    url = (
        f"https://api.apify.com/v2/actor-runs/{run_id}/resurrect"
        f"?timeout={timeout_secs}&memory={memory_mbytes}"
    )
    req = urllib.request.Request(
        url,
        method="POST",
        headers={"Authorization": f"Bearer {token}"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8")).get("data") or {}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", help="Resurrect a specific run id")
    parser.add_argument(
        "--max-items-threshold",
        type=int,
        default=50,
        help="Only auto-pick TIMED-OUT runs with fewer than this many items",
    )
    parser.add_argument("--limit", type=int, default=2, help="Max runs to resurrect")
    parser.add_argument("--timeout", type=int, default=3600, help="Desired timeout secs (API only)")
    parser.add_argument("--memory", type=int, default=2048)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--prefer-api", action="store_true")
    args = parser.parse_args()

    if args.run_id:
        candidates = [(item_count(""), {"id": args.run_id, "defaultDatasetId": None})]
        # fix count
        # will fetch below
        runs = list_runs(50)
        match = next((r for r in runs if r.get("id") == args.run_id), {"id": args.run_id})
        n = item_count(match.get("defaultDatasetId") or "")
        candidates = [(n, match)]
    else:
        runs = list_runs(50)
        running = [r for r in runs if r.get("status") in ("RUNNING", "READY")]
        print(f"Currently RUNNING/READY: {len(running)}")
        for r in running:
            print(f"  {r.get('status')} {r.get('id')} {r.get('statusMessage')}")

        timed = []
        for r in runs:
            if r.get("status") != "TIMED-OUT":
                continue
            n = item_count(r.get("defaultDatasetId") or "")
            timed.append((n, r))
            print(f"TIMED-OUT {r.get('id')} items={n}")
        timed.sort(key=lambda x: x[0] if x[0] >= 0 else 9999)
        candidates = [
            (n, r) for n, r in timed if 0 <= n < args.max_items_threshold
        ][: args.limit]

    print(f"\nResurrect candidates: {len(candidates)}")
    for n, r in candidates:
        print(f"  {r.get('id')} items={n}")

    if args.dry_run or not candidates:
        return 0

    ok = 0
    for n, r in candidates:
        rid = r["id"]
        print(f"\nResurrecting {rid} (had {n} items) …", flush=True)
        try:
            if args.prefer_api:
                data = resurrect_api(rid, args.timeout, args.memory)
            else:
                try:
                    data = resurrect_api(rid, args.timeout, args.memory)
                except Exception as api_exc:
                    print(f"  API longer-timeout failed ({api_exc}); using CLI resurrect", flush=True)
                    data = resurrect_cli(rid)
            print(
                f"  status={data.get('status')} msg={data.get('statusMessage')} "
                f"timeout={(data.get('options') or {}).get('timeoutSecs')}",
                flush=True,
            )
            ok += 1
            time.sleep(5)
        except Exception as exc:  # noqa: BLE001
            print(f"  FAILED: {exc}", file=sys.stderr, flush=True)
            if "MEMORY" in str(exc).upper():
                print("Stopping: free memory by waiting for RUNNING jobs.", flush=True)
                break
    print(f"\nResurrected {ok}/{len(candidates)}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
