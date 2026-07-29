"""
Download Apify datasets listed in the fire-and-forget manifest into data/raw/.

Merges with existing local dumps; updates seen_post_ids.json for dedup stats.

  python scripts/02_download_datasets.py
  python scripts/02_download_datasets.py --manifest data/raw/apify_fire_forget_manifest.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from seen_ids import SeenIdRegistry, normalize_post_id  # noqa: E402

DEFAULT_MANIFEST = ROOT / "data" / "raw" / "apify_fire_forget_manifest.json"


def resolve_apify_cmd() -> list[str]:
    try:
        from importlib.util import module_from_spec, spec_from_file_location

        spec = spec_from_file_location("scrape01", ROOT / "scripts" / "01_scrape_apify.py")
        mod = module_from_spec(spec)
        assert spec.loader
        spec.loader.exec_module(mod)
        return mod.resolve_apify_cmd()
    except Exception:  # noqa: BLE001
        return ["apify"]


def resolve_token() -> str | None:
    return os.environ.get("APIFY_TOKEN") or os.environ.get("APIFY_API_TOKEN")


def parse_json_items(text: str) -> list:
    text = (text or "").strip()
    if not text:
        return []
    if text[0] not in "[{":
        for i, ch in enumerate(text):
            if ch in "[{":
                text = text[i:]
                break
        else:
            return []
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, list) else [parsed]
    except json.JSONDecodeError:
        return []


def download_dataset(dataset_id: str) -> list:
    token = resolve_token()
    if token:
        from apify_client import ApifyClient

        client = ApifyClient(token)
        return list(client.dataset(dataset_id).iterate_items())

    cmd = [
        *resolve_apify_cmd(),
        "datasets",
        "get-items",
        dataset_id,
        "--format",
        "json",
    ]
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    items = parse_json_items(proc.stdout or "")
    if not items:
        items = parse_json_items(f"{proc.stderr or ''}\n{proc.stdout or ''}")
    if proc.returncode != 0 and not items:
        raise RuntimeError(
            f"download {dataset_id} failed: {(proc.stderr or proc.stdout or '')[:1000]}"
        )
    return items


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Fire-and-forget manifest JSON",
    )
    parser.add_argument(
        "--only-succeeded",
        action="store_true",
        help="Skip entries that already have local path written by this script",
    )
    args = parser.parse_args()

    if not args.manifest.is_file():
        print(f"No manifest at {args.manifest}", file=sys.stderr)
        return 1

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    started = manifest.get("started") or []
    registry = SeenIdRegistry(seed_from_dumps=True)
    print(f"Registry before download: {len(registry)} IDs", flush=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    total_raw = 0
    total_new = 0
    total_dup = 0
    downloaded = 0

    for entry in started:
        if entry.get("error") or not entry.get("dataset_id"):
            print(f"SKIP (no dataset): {entry.get('run_name')}", flush=True)
            continue
        run_name = entry["run_name"]
        corpus = entry.get("corpus") or "a"
        dataset_id = entry["dataset_id"]
        out_dir = ROOT / "data" / "raw" / f"corpus_{corpus}"
        out_path = out_dir / f"{run_name}_apify_{dataset_id}_{stamp}.json"

        print(f"\nDownload {run_name} dataset={dataset_id} …", flush=True)
        try:
            items = download_dataset(dataset_id)
        except Exception as exc:  # noqa: BLE001
            print(f"  FAILED: {exc}", file=sys.stderr, flush=True)
            entry["download_error"] = str(exc)[:1000]
            continue

        new_items, n_new, n_dup = registry.filter_new(items)
        total_raw += len(items)
        total_new += n_new
        total_dup += n_dup
        meta = {
            "corpus": corpus,
            "run_name": run_name,
            "actor_id": entry.get("actor_id"),
            "apify_run_id": entry.get("run_id"),
            "dataset_id": dataset_id,
            "scraped_at": stamp,
            "source": "apify_download",
            "n_items_raw": len(items),
            "n_items_new": n_new,
            "n_items_dup": n_dup,
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps({"meta": meta, "items": items}, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        entry["local_path"] = str(out_path)
        entry["n_items_raw"] = len(items)
        entry["n_items_new"] = n_new
        entry["n_items_dup"] = n_dup
        downloaded += 1
        print(f"  raw={len(items)} new={n_new} dup={n_dup} → {out_path.name}", flush=True)
        registry.save()

    # write updated manifest
    manifest["downloaded_at"] = stamp
    manifest["download_summary"] = {
        "datasets": downloaded,
        "total_raw": total_raw,
        "total_new": total_new,
        "total_dup": total_dup,
        "registry_size": len(registry),
    }
    args.manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        f"\nDone. datasets={downloaded} raw={total_raw} new={total_new} "
        f"dup={total_dup} registry={len(registry)}",
        flush=True,
    )
    print("Next: python scripts/03_filter_and_normalize.py", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
