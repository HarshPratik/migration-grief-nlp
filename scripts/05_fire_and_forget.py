"""
Fire-and-forget: start ONLY unfinished scrape jobs on Apify, then exit.

- Skips run_names that already have a solid local dump (>= --min-items posts)
- Retries thin dumps (e.g. 1-item timeouts) so we don't waste full re-runs of good batches
- Uses non-blocking `actors start` — you can close the laptop after this script finishes
- Writes a manifest of run_id + dataset_id for later download

Examples:
  python scripts/05_fire_and_forget.py --dry-run
  python scripts/05_fire_and_forget.py --min-items 10 --stagger-seconds 15
  python scripts/05_fire_and_forget.py --max-starts 20
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from scrape_plan import unfinished_runs  # noqa: E402

# Reuse CLI resolver from 01 if available
try:
    from importlib.util import module_from_spec, spec_from_file_location

    spec = spec_from_file_location("scrape01", ROOT / "scripts" / "01_scrape_apify.py")
    mod = module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(mod)
    resolve_apify_cmd = mod.resolve_apify_cmd
    resolve_token = mod.resolve_token
except Exception:  # noqa: BLE001

    def resolve_token():
        return os.environ.get("APIFY_TOKEN") or os.environ.get("APIFY_API_TOKEN")

    def resolve_apify_cmd():
        return ["apify"]


MANIFEST_DIR = ROOT / "data" / "raw"
DEFAULT_MANIFEST = MANIFEST_DIR / "apify_fire_forget_manifest.json"


def start_via_client(
    actor_id: str,
    run_input: dict,
    timeout_secs: int,
    memory_mbytes: int | None = 1024,
):
    from apify_client import ApifyClient

    token = resolve_token()
    if not token:
        raise RuntimeError("APIFY_TOKEN required for apify-client path")
    client = ApifyClient(token)
    kwargs = {"run_input": run_input, "timeout_secs": timeout_secs}
    if memory_mbytes:
        kwargs["memory_mbytes"] = memory_mbytes
    try:
        run = client.actor(actor_id).start(**kwargs)
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)
        if "memory limit" in msg.lower():
            raise RuntimeError(f"APIFY_MEMORY_LIMIT: {msg}") from exc
        raise
    return {
        "run_id": run.get("id"),
        "dataset_id": run.get("defaultDatasetId"),
        "status": run.get("status"),
        "via": "client",
    }


def _parse_json_blob(text: str) -> dict | None:
    text = (text or "").strip()
    if not text:
        return None
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        i = text.find("{")
        if i < 0:
            return None
        try:
            obj = json.loads(text[i:])
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None


def _extract_run_meta(payload: dict) -> dict:
    """Normalize Apify CLI / API start responses into run_id + dataset_id."""
    # Shape A: { ok, run: { id, status }, ... }
    run = payload.get("run") if isinstance(payload.get("run"), dict) else None
    if run and run.get("id"):
        return {
            "run_id": run.get("id"),
            "dataset_id": run.get("defaultDatasetId")
            or payload.get("defaultDatasetId"),
            "status": run.get("status"),
            "raw": payload,
        }
    # Shape B: { data: { id, defaultDatasetId, status } }
    data = payload.get("data") if isinstance(payload.get("data"), dict) else payload
    return {
        "run_id": data.get("id") or data.get("runId"),
        "dataset_id": data.get("defaultDatasetId") or data.get("defaultDatasetID"),
        "status": data.get("status"),
        "raw": payload,
    }


def start_via_cli(
    actor_id: str,
    run_input: dict,
    timeout_secs: int,
    memory_mbytes: int = 1024,
) -> dict:
    fd, input_path = tempfile.mkstemp(suffix=".json", prefix="apify_ff_")
    os.close(fd)
    Path(input_path).write_text(json.dumps(run_input), encoding="utf-8")
    try:
        cmd = [
            *resolve_apify_cmd(),
            "actors",
            "start",
            actor_id,
            "--input-file",
            input_path,
            "--json",
            "-t",
            str(timeout_secs),
            "-m",
            str(memory_mbytes),
        ]
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        out = (proc.stdout or "").strip()
        err = (proc.stderr or "").strip()
        combined = f"{out}\n{err}"

        # Memory / plan hard failures
        if "exceed the memory limit" in combined or "memory limit" in combined.lower():
            raise RuntimeError(
                "APIFY_MEMORY_LIMIT: account concurrent actor memory is full "
                f"(16384MB cap). Wait for running jobs to finish or abort them in "
                f"Console, then retry with --memory 1024 and --max-starts 1. "
                f"Detail: {combined[:800]}"
            )

        payload = _parse_json_blob(out) or _parse_json_blob(err)
        if payload:
            # CLI sometimes exits non-zero even when start succeeded
            if payload.get("ok") is True or payload.get("run") or payload.get("id"):
                meta = _extract_run_meta(payload)
                if meta.get("run_id"):
                    meta["via"] = "cli"
                    return meta

        if proc.returncode != 0:
            raise RuntimeError(
                f"actors start failed code={proc.returncode}: {combined[:1500]}"
            )
        raise RuntimeError(f"Could not parse start response: {combined[:500]}")
    finally:
        try:
            os.unlink(input_path)
        except OSError:
            pass


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Start unfinished Apify scrape jobs and exit (fire-and-forget)"
    )
    parser.add_argument("--corpus", choices=["a", "b", "both"], default="both")
    parser.add_argument(
        "--min-items",
        type=int,
        default=10,
        help="Local dump with >= this many items = skip (avoid re-paying for good runs)",
    )
    parser.add_argument(
        "--max-starts",
        type=int,
        default=0,
        help="Cap how many new jobs to start (0 = all unfinished)",
    )
    parser.add_argument(
        "--stagger-seconds",
        type=float,
        default=90.0,
        help="Pause between starts (default 90s). Account has 16GB concurrent memory cap.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Per-actor timeout seconds on Apify (default 3600)",
    )
    parser.add_argument(
        "--memory",
        type=int,
        default=1024,
        help="MB per actor run (default 1024; lower = more concurrent slots under 16GB cap)",
    )
    parser.add_argument(
        "--stop-on-memory-limit",
        action="store_true",
        default=True,
        help="Stop queue when Apify concurrent memory is full (default on)",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--prefer-cli",
        action="store_true",
        help="Use `apify actors start` even if APIFY_TOKEN is set",
    )
    args = parser.parse_args()

    todo, skipped = unfinished_runs(
        corpus=args.corpus, mode="full", min_items=args.min_items
    )
    print(f"Completed (skip — protect budget): {len(skipped)}", flush=True)
    for s in skipped[:15]:
        print(f"  SKIP {s['run_name']} ({s['n_items']} items)", flush=True)
    if len(skipped) > 15:
        print(f"  ... +{len(skipped) - 15} more skips", flush=True)

    print(f"\nUnfinished to start: {len(todo)}", flush=True)
    for row in todo:
        lab, name, _rin, _act, reason = row
        extra = f" [{reason}]" if reason else ""
        print(f"  TODO corpus_{lab}/{name}{extra}", flush=True)

    if args.max_starts and args.max_starts > 0:
        todo = todo[: args.max_starts]
        print(f"\nCapped to first {len(todo)} starts (--max-starts)", flush=True)

    if args.dry_run:
        print("\nDry run — nothing started on Apify.", flush=True)
        return 0

    if not todo:
        print("\nNothing to start. All planned search runs already have solid local dumps.", flush=True)
        return 0

    use_cli = args.prefer_cli or not resolve_token()
    print(
        f"\nStarting {len(todo)} Apify jobs "
        f"({'CLI actors start' if use_cli else 'apify-client start'}) "
        f"memory={args.memory}MB stagger={args.stagger_seconds}s …",
        flush=True,
    )
    print(
        "NOTE: Plan concurrent memory cap is often 16384MB. "
        "If many runs are already RUNNING, new starts fail until they finish.",
        flush=True,
    )
    print("You can close the laptop after this script prints DONE.", flush=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    entries = []
    ok = 0
    fail = 0

    for i, (lab, name, rin, actor_id, reason) in enumerate(todo, start=1):
        print(f"\n[{i}/{len(todo)}] START {name} …", flush=True)
        try:
            if use_cli:
                meta = start_via_cli(
                    actor_id, rin, args.timeout, memory_mbytes=args.memory
                )
            else:
                meta = start_via_client(
                    actor_id, rin, args.timeout, memory_mbytes=args.memory
                )
            entry = {
                "corpus": lab,
                "run_name": name,
                "actor_id": actor_id,
                "run_id": meta.get("run_id"),
                "dataset_id": meta.get("dataset_id"),
                "status": meta.get("status"),
                "started_at": stamp,
                "retry_reason": reason,
                "input": rin,
                "via": meta.get("via"),
                "memory_mbytes": args.memory,
            }
            entries.append(entry)
            ok += 1
            print(
                f"  OK run_id={entry['run_id']} dataset_id={entry['dataset_id']} "
                f"status={entry['status']}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            fail += 1
            err = str(exc)
            print(f"  FAILED: {err}", file=sys.stderr, flush=True)
            entries.append(
                {
                    "corpus": lab,
                    "run_name": name,
                    "actor_id": actor_id,
                    "error": err[:2000],
                    "started_at": stamp,
                }
            )
            if args.stop_on_memory_limit and "APIFY_MEMORY_LIMIT" in err:
                print(
                    "\nStopping further starts: concurrent memory full. "
                    "Wait for running Apify jobs to finish, then re-run this script "
                    "(completed dumps still skipped).",
                    flush=True,
                )
                # still write manifest below
                manifest = {
                    "created_at": stamp,
                    "min_items_skip_threshold": args.min_items,
                    "skipped_completed": skipped,
                    "started": entries,
                    "summary": {
                        "ok": ok,
                        "fail": fail,
                        "skipped": len(skipped),
                        "stopped_reason": "memory_limit",
                    },
                }
                MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
                DEFAULT_MANIFEST.write_text(
                    json.dumps(manifest, indent=2), encoding="utf-8"
                )
                (MANIFEST_DIR / f"apify_fire_forget_manifest_{stamp}.json").write_text(
                    json.dumps(manifest, indent=2), encoding="utf-8"
                )
                break

        # persist after each start so a mid-crash still keeps IDs
        manifest = {
            "created_at": stamp,
            "min_items_skip_threshold": args.min_items,
            "skipped_completed": skipped,
            "started": entries,
            "summary": {"ok": ok, "fail": fail, "skipped": len(skipped)},
        }
        MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
        DEFAULT_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        (MANIFEST_DIR / f"apify_fire_forget_manifest_{stamp}.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )

        if i < len(todo) and args.stagger_seconds > 0:
            time.sleep(args.stagger_seconds)

    print(
        f"\nDONE. Started ok={ok} fail={fail} | skipped_completed={len(skipped)}",
        flush=True,
    )
    print(f"Manifest → {DEFAULT_MANIFEST}", flush=True)
    print(
        "\nJobs are running on Apify. You may close this laptop.\n"
        "Later download:\n"
        "  python scripts/02_download_datasets.py\n"
        "Console: https://console.apify.com/actors/runs\n",
        flush=True,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
