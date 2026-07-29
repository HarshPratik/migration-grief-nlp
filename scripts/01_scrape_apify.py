"""
Phase 1 — Scrape Reddit posts via Apify (budget-aware).

Full mode strategy:
  1) trudax/reddit-scraper-lite SEARCH runs (higher volume per $)
  2) labrat011/reddit-scraper listing/search (reliable, often 25/run free-tier)

Stops with exit code 42 when Apify balance/credits are exhausted.
Resume with --resume after logging into a new account.

Examples:
  python scripts/01_scrape_apify.py --mode full --corpus both
  python scripts/01_scrape_apify.py --mode full --corpus both --resume
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from seen_ids import SeenIdRegistry  # noqa: E402

CONFIG_PATH = ROOT / "config" / "scrape_runs.yaml"
FILTERS_PATH = ROOT / "config" / "filters.yaml"

BUDGET_EXIT_CODE = 42

# True payment / platform credit exhaustion (stop campaign).
# Keep this VERY strict — actor logs contain words like "limit"/"usage" that
# must NOT stop the whole campaign.
BUDGET_PATTERNS = re.compile(
    r"("
    r"insufficient\s+(funds|credit|credits|balance)|"
    r"out\s+of\s+(credit|credits|money|funds)|"
    r"not\s+enough\s+(credit|credits|funds|money|balance)|"
    r"payment\s+required|"
    r"monthly\s+usage\s+hard\s+limit|"
    r"you\s+have\s+run\s+out\s+of\s+(credit|credits|prepaid)|"
    r"please\s+upgrade\s+your\s+subscription|"
    r"please\s+top\s+up|"
    r"402\s+payment"
    r")",
    re.I,
)

# Plan concurrency / queue limits — reduce workers, do NOT treat as empty wallet
CONCURRENCY_PATTERNS = re.compile(
    r"("
    r"concurrent\s+runs?|"
    r"too\s+many\s+parallel|"
    r"maximum\s+number\s+of\s+(concurrent\s+)?runs?|"
    r"you\s+have\s+reached\s+the\s+maximum|"
    r"memory\s+limit\s+exceeded|"
    r"actor\s+is\s+already\s+running"
    r")",
    re.I,
)


class BudgetExhaustedError(RuntimeError):
    pass


def load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_token() -> str | None:
    token = os.environ.get("APIFY_TOKEN") or os.environ.get("APIFY_API_TOKEN")
    if token:
        return token
    for p in (
        Path.home() / ".apify" / "auth.json",
        Path(os.environ.get("APPDATA", "")) / "apify-cli" / "auth.json",
    ):
        if p.is_file():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                t = data.get("token") or data.get("apiToken")
                if t:
                    return t
            except (json.JSONDecodeError, OSError):
                continue
    return None


def looks_like_budget_error(text: str) -> bool:
    if not text:
        return False
    # Concurrency limits are not empty-balance
    if CONCURRENCY_PATTERNS.search(text):
        return False
    return bool(BUDGET_PATTERNS.search(text))


def looks_like_concurrency_error(text: str) -> bool:
    return bool(text and CONCURRENCY_PATTERNS.search(text))


def build_trudax_runs(
    label: str,
    subreddits: list[str],
    queries: list[str],
    cfg: dict,
    limits: dict,
) -> list[tuple[str, str, dict]]:
    """(corpus, run_name, input) for trudax search + listing URLs."""
    runs: list[tuple[str, str, dict]] = []
    base = dict(cfg.get("trudax") or {})
    max_items = int(limits.get("maxItems", 1000))
    max_posts = int(limits.get("maxPostCount", 500))

    # Batch search queries (fewer actor starts → less overhead; still good coverage)
    batch_size = 5
    for sub in subreddits:
        for bi in range(0, len(queries), batch_size):
            batch = queries[bi : bi + batch_size]
            name = f"trudax_{sub}_search_batch{bi // batch_size}"
            # sort/time variants expand coverage past Reddit list caps
            for sort, time_f in (("new", "year"), ("top", "year"), ("relevance", "all")):
                rin = {
                    **base,
                    "searches": batch,
                    "searchCommunityName": sub,
                    "searchPosts": True,
                    "sort": sort,
                    "time": time_f,
                    "maxItems": max_items,
                    "maxPostCount": max_posts,
                    "maxComments": 0,
                }
                runs.append((label, f"{name}_{sort}_{time_f}", rin))

        # Listing startUrls (often thin under Reddit anti-bot; still useful)
        for sort in cfg.get("listing_sorts") or ["new", "top", "hot"]:
            if sort == "top":
                for tf in cfg.get("time_filters_for_top") or ["year"]:
                    url = f"https://www.reddit.com/r/{sub}/top/?t={tf}"
                    name = f"trudax_{sub}_top_{tf}"
                    rin = {
                        **base,
                        "startUrls": [{"url": url}],
                        "maxItems": max_items,
                        "maxPostCount": max_posts,
                        "maxComments": 0,
                    }
                    runs.append((label, name, rin))
            else:
                url = f"https://www.reddit.com/r/{sub}/{sort}/"
                name = f"trudax_{sub}_{sort}"
                rin = {
                    **base,
                    "startUrls": [{"url": url}],
                    "maxItems": max_items,
                    "maxPostCount": max_posts,
                    "maxComments": 0,
                }
                runs.append((label, name, rin))
    return runs


def build_labrat_runs(
    label: str,
    subreddits: list[str],
    queries: list[str],
    cfg: dict,
    limits: dict,
) -> list[tuple[str, str, dict]]:
    runs: list[tuple[str, str, dict]] = []
    base = dict(cfg.get("labrat") or {})
    max_results = int(limits.get("maxResults", 1000))

    for sub in subreddits:
        for sort in cfg.get("listing_sorts") or ["new", "top", "hot"]:
            if sort == "top":
                for tf in cfg.get("time_filters_for_top") or ["year"]:
                    name = f"labrat_{sub}_top_{tf}"
                    rin = {
                        **base,
                        "mode": "subreddit_posts",
                        "subreddits": [sub],
                        "sort": "top",
                        "timeFilter": tf,
                        "maxResults": max_results,
                        "includeComments": False,
                    }
                    runs.append((label, name, rin))
            else:
                name = f"labrat_{sub}_{sort}"
                rin = {
                    **base,
                    "mode": "subreddit_posts",
                    "subreddits": [sub],
                    "sort": sort,
                    "maxResults": max_results,
                    "includeComments": False,
                }
                runs.append((label, name, rin))

        # Batch search
        name = f"labrat_{sub}_search_batch"
        rin = {
            **base,
            "mode": "search",
            "searchSubreddit": sub,
            "searchQueriesList": queries,
            "searchSort": "new",
            "timeFilter": "year",
            "maxResults": max_results,
            "includeComments": False,
        }
        runs.append((label, name, rin))
    return runs


def interleave_runs(
    planned: list[tuple[str, str, dict]],
) -> list[tuple[str, str, dict]]:
    by_label: dict[str, list] = {}
    for item in planned:
        by_label.setdefault(item[0], []).append(item)
    labels = sorted(by_label.keys())
    out: list = []
    idx = {lb: 0 for lb in labels}
    while True:
        progressed = False
        for lb in labels:
            i = idx[lb]
            if i < len(by_label[lb]):
                out.append(by_label[lb][i])
                idx[lb] = i + 1
                progressed = True
        if not progressed:
            break
    return out


def save_items(items: list, out_path: Path, meta: dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"meta": meta, "items": items}
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"  Saved {len(items)} items → {out_path}", flush=True)


def parse_json_items(stdout: str) -> list:
    stdout = (stdout or "").strip()
    if not stdout:
        return []
    if stdout[0] not in "[{":
        for i, ch in enumerate(stdout):
            if ch in "[{":
                stdout = stdout[i:]
                break
        else:
            return []
    try:
        parsed = json.loads(stdout)
        return parsed if isinstance(parsed, list) else [parsed]
    except json.JSONDecodeError:
        items: list = []
        for line in stdout.splitlines():
            line = line.strip()
            if not line or line[0] not in "{[":
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, list):
                items.extend(obj)
            else:
                items.append(obj)
        return items


def resolve_apify_cmd() -> list[str]:
    """Return argv prefix that invokes Apify CLI on Windows/Unix.

    Prefer `node …/apify.js` so subprocess does not need a shell / .ps1 shim.
    """
    import shutil

    node = shutil.which("node") or shutil.which("node.exe")
    js_candidates = [
        Path(r"C:\nvm4w\nodejs\node_modules\apify-cli\dist\apify.js"),
        Path(os.environ.get("APPDATA", ""))
        / "npm"
        / "node_modules"
        / "apify-cli"
        / "dist"
        / "apify.js",
    ]
    # Also relative to node binary
    if node:
        node_dir = Path(node).resolve().parent
        js_candidates.insert(0, node_dir / "node_modules" / "apify-cli" / "dist" / "apify.js")

    if node:
        for js in js_candidates:
            if js.is_file():
                return [node, str(js)]

    for name in ("apify.cmd", "apify.exe", "apify"):
        path = shutil.which(name)
        if path:
            return [path]
    return ["apify"]


def _download_dataset(dataset_id: str) -> list:
    """Fetch dataset items via Apify CLI (works after timeout/abort)."""
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
    combined = f"{proc.stderr or ''}\n{proc.stdout or ''}"
    if looks_like_budget_error(combined):
        raise BudgetExhaustedError(combined[:3000])
    items = parse_json_items(proc.stdout or "")
    if not items:
        # stderr may contain Content-Type noise then JSON
        items = parse_json_items(combined)
    return items


def _extract_dataset_id(text: str) -> str | None:
    m = re.search(r"Dataset ID:\s*([A-Za-z0-9]+)", text or "")
    if m:
        return m.group(1)
    m = re.search(r"datasets/([A-Za-z0-9]+)", text or "")
    if m:
        return m.group(1)
    return None


def run_actor_via_cli(
    actor_id: str, run_input: dict, timeout_secs: int = 3600
) -> tuple[list, dict]:
    # write then close before subprocess reads it (Windows lock)
    fd, input_path = tempfile.mkstemp(suffix=".json", prefix="apify_in_")
    os.close(fd)
    Path(input_path).write_text(json.dumps(run_input), encoding="utf-8")
    try:
        # Do NOT use --silent: we need Dataset ID on timeout to pull partial results.
        # Prefer --output-dataset for success path; still parse logs on failure.
        cmd = [
            *resolve_apify_cmd(),
            "call",
            actor_id,
            "-f",
            input_path,
            "--output-dataset",
            "-t",
            str(timeout_secs),
        ]
        print(f"  CLI: apify call {actor_id} (timeout={timeout_secs}s) …", flush=True)
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            shell=False,
        )
        combined = f"{proc.stderr or ''}\n{proc.stdout or ''}"
        # Only treat as budget if message is clearly payment-related AND
        # we got no useful output (avoid false positives from crawler logs).
        if looks_like_budget_error(combined) and "payment" in combined.lower() or (
            looks_like_budget_error(combined)
            and any(
                k in combined.lower()
                for k in (
                    "insufficient funds",
                    "out of credit",
                    "not enough credit",
                    "payment required",
                    "hard limit",
                    "run out of credit",
                    "top up",
                )
            )
        ):
            raise BudgetExhaustedError(combined[:3000])

        items = parse_json_items(proc.stdout or "")
        dataset_id = _extract_dataset_id(combined)
        run_id_m = re.search(r"Run ID:\s*([A-Za-z0-9]+)", combined)
        run_id = run_id_m.group(1) if run_id_m else "cli"

        # Timeout/abort often returns code != 0 but still has a dataset
        if (not items) and dataset_id:
            print(f"  Pulling dataset {dataset_id} (partial/timeout recovery) …", flush=True)
            try:
                items = _download_dataset(dataset_id)
            except BudgetExhaustedError:
                raise
            except Exception as exc:  # noqa: BLE001
                print(f"  Dataset pull failed: {exc}", flush=True)

        # Local CLI killed / node exit 13 while actor may still run or have data
        if proc.returncode != 0 and not items:
            if looks_like_budget_error(combined) and any(
                k in combined.lower()
                for k in (
                    "insufficient funds",
                    "out of credit",
                    "not enough credit",
                    "payment required",
                    "hard limit",
                    "run out of credit",
                    "top up",
                )
            ):
                raise BudgetExhaustedError(combined[:3000])
            # Try resurrect if timed out and we have a run id
            if run_id and run_id != "cli" and (
                "TIMED-OUT" in combined or "timeout" in combined.lower()
            ):
                print(f"  Attempting resurrect {run_id} …", flush=True)
                try:
                    subprocess.run(
                        [*resolve_apify_cmd(), "runs", "resurrect", run_id, "--json"],
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        check=False,
                    )
                    if dataset_id:
                        items = _download_dataset(dataset_id)
                except Exception as exc:  # noqa: BLE001
                    print(f"  Resurrect/pull failed: {exc}", flush=True)
            if not items:
                raise RuntimeError(
                    f"apify call failed (code={proc.returncode}): "
                    f"{combined[-2000:]}"
                )

        run_meta = {
            "id": run_id,
            "defaultDatasetId": dataset_id or "cli",
            "status": "SUCCEEDED" if proc.returncode == 0 else "FAILED_OR_PARTIAL",
            "returncode": proc.returncode,
            "stderr_tail": (proc.stderr or "")[-800:],
        }
        print(
            f"  CLI finished — {len(items)} items (code={proc.returncode})",
            flush=True,
        )
        return items, run_meta
    finally:
        try:
            os.unlink(input_path)
        except OSError:
            pass


def run_actor_client(client, actor_id: str, run_input: dict, timeout_secs: int = 3600):
    print(f"  Calling actor {actor_id} (client) …", flush=True)
    try:
        run = client.actor(actor_id).call(run_input=run_input, timeout_secs=timeout_secs)
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)
        if looks_like_budget_error(msg):
            raise BudgetExhaustedError(msg) from exc
        raise
    items = list(client.dataset(run["defaultDatasetId"]).iterate_items())
    print(
        f"  Run {run['id']} finished — {len(items)} items (status={run.get('status')})",
        flush=True,
    )
    return items, run


def already_done(corpus_dir: Path, run_name: str, min_items: int = 10) -> Path | None:
    """Treat as done only if a dump has >= min_items (retry thin timeout dumps)."""
    if not corpus_dir.is_dir():
        return None
    best_path = None
    best_n = 0
    for path in sorted(corpus_dir.glob(f"{run_name}_*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            n = len(data.get("items") or [])
        except (json.JSONDecodeError, OSError):
            continue
        if n > best_n:
            best_n = n
            best_path = path
    if best_path is not None and best_n >= min_items:
        return best_path
    return None


def execute_one_run(
    *,
    label: str,
    name: str,
    rin: dict,
    actor_id: str,
    mode: str,
    stamp: str,
    timeout: int,
    use_cli: bool,
    client,
    registry: SeenIdRegistry,
    print_lock: threading.Lock,
) -> dict:
    """Run one actor job; return log row dict. Raises BudgetExhaustedError."""
    t0 = time.time()
    corpus_dir = ROOT / "data" / "raw" / (
        "pilot" if mode == "pilot" else f"corpus_{label}"
    )
    if mode == "pilot":
        out = ROOT / "data" / "raw" / "pilot" / f"corpus_{label}_{name}.json"
    else:
        out = corpus_dir / f"{name}_{stamp}.json"

    with print_lock:
        print(f"\n>>> START corpus {label.upper()} | {name} [{actor_id}]", flush=True)

    if use_cli:
        items, run = run_actor_via_cli(actor_id, rin, timeout_secs=timeout)
    else:
        items, run = run_actor_client(client, actor_id, rin, timeout_secs=timeout)

    elapsed = time.time() - t0
    raw_n = len(items)
    if raw_n == 0:
        with print_lock:
            print(f"<<< EMPTY {name} ({elapsed:.0f}s)", flush=True)
        return {
            "corpus": label,
            "run_name": name,
            "status": "empty",
            "n_items": 0,
            "n_raw": 0,
            "n_new": 0,
            "n_dup": 0,
            "seconds": round(elapsed, 1),
            "actor_id": actor_id,
        }

    new_items, n_new, n_dup = registry.filter_new(items)
    meta = {
        "corpus": label,
        "run_name": name,
        "mode": mode,
        "actor_id": actor_id,
        "apify_run_id": run.get("id"),
        "dataset_id": run.get("defaultDatasetId"),
        "scraped_at": stamp,
        "input": rin,
        "n_items_raw": raw_n,
        "n_items_new": n_new,
        "n_items_dup": n_dup,
        "seconds": round(elapsed, 1),
    }
    # Always save raw for audit; filter pipeline dedupes again later.
    # Prefer saving NEW-only to save disk when all dups — still save raw_n for resume skip.
    save_items(items, out, meta)
    registry.save()

    sec_per_post = (elapsed / raw_n) if raw_n else None
    with print_lock:
        print(
            f"<<< DONE {name}: raw={raw_n} new={n_new} dup={n_dup} "
            f"in {elapsed:.0f}s ({sec_per_post:.1f}s/post)"
            if sec_per_post
            else f"<<< DONE {name}: raw={raw_n} new={n_new} dup={n_dup} in {elapsed:.0f}s",
            flush=True,
        )
    return {
        "corpus": label,
        "run_name": name,
        "status": "ok",
        "actor_id": actor_id,
        "n_items": raw_n,
        "n_raw": raw_n,
        "n_new": n_new,
        "n_dup": n_dup,
        "seconds": round(elapsed, 1),
        "sec_per_post": round(sec_per_post, 2) if sec_per_post else None,
        "path": str(out),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Scrape Reddit via Apify for Phase 1")
    parser.add_argument("--mode", choices=["pilot", "full"], default="pilot")
    parser.add_argument("--corpus", choices=["a", "b", "both"], default="both")
    parser.add_argument(
        "--engine",
        choices=["trudax", "labrat", "both"],
        default="both",
        help="Which actor family to use (full mode)",
    )
    parser.add_argument("--listings-only", action="store_true")
    parser.add_argument("--searches-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Per-run timeout seconds (default 3600; 1200 often times out mid-crawl)",
    )
    parser.add_argument("--max-runs", type=int, default=0)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel Apify actor runs (1 recommended on 16GB memory plan)",
    )
    parser.add_argument(
        "--resurrect-on-timeout",
        action="store_true",
        help="If a run times out, try `apify runs resurrect` once with same queue",
    )
    args = parser.parse_args()

    cfg = load_yaml(CONFIG_PATH)
    filters = load_yaml(FILTERS_PATH)
    limits = dict(cfg[args.mode])
    primary = cfg.get("primary_actor_id") or "trudax/reddit-scraper-lite"
    secondary = cfg.get("secondary_actor_id") or "labrat011/reddit-scraper"

    corpora = []
    if args.corpus in ("a", "both"):
        corpora.append(
            ("a", filters["subreddits"]["corpus_a"], cfg["search_queries"]["corpus_a"])
        )
    if args.corpus in ("b", "both"):
        corpora.append(
            ("b", filters["subreddits"]["corpus_b"], cfg["search_queries"]["corpus_b"])
        )

    planned: list[tuple[str, str, dict, str]] = []
    for label, subs, queries in corpora:
        if args.engine in ("trudax", "both"):
            for lab, name, rin in build_trudax_runs(label, subs, queries, cfg, limits):
                if args.listings_only and "search" in name:
                    continue
                if args.searches_only and "search" not in name:
                    continue
                planned.append((lab, name, rin, primary))
        if args.engine in ("labrat", "both"):
            for lab, name, rin in build_labrat_runs(label, subs, queries, cfg, limits):
                if args.listings_only and "search" in name:
                    continue
                if args.searches_only and "search" not in name:
                    continue
                planned.append((lab, name, rin, secondary))

    planned_simple = [(a, b, c) for a, b, c, _ in planned]
    order = interleave_runs(planned_simple)
    actor_map = {(lab, name): act for lab, name, _, act in planned}
    planned_full = [(lab, name, rin, actor_map[(lab, name)]) for lab, name, rin in order]

    if args.max_runs and args.max_runs > 0:
        planned_full = planned_full[: args.max_runs]

    workers = max(1, int(args.workers))
    print(
        f"Mode={args.mode} | engine={args.engine} | workers={workers} | "
        f"planned runs={len(planned_full)}",
        flush=True,
    )
    for lab, name, _, act in planned_full[:30]:
        print(f"  - corpus_{lab}/{name}  [{act}]", flush=True)
    if len(planned_full) > 30:
        print(f"  ... +{len(planned_full) - 30} more", flush=True)

    if args.dry_run:
        print("Dry run — no Apify calls.", flush=True)
        return 0

    token = resolve_token()
    client = None
    use_cli = True
    if token:
        from apify_client import ApifyClient

        client = ApifyClient(token)
        use_cli = False
        print("Auth: APIFY_TOKEN (apify-client)", flush=True)
    else:
        print("Auth: Apify CLI keyring (`apify call`)", flush=True)

    registry = SeenIdRegistry(seed_from_dumps=True)
    print(f"Dedup registry seeded with {len(registry)} known post IDs", flush=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_rows: list[dict] = []
    log_lock = threading.Lock()
    print_lock = threading.Lock()
    total_raw = 0
    total_new = 0
    total_dup = 0
    budget_hit = False
    budget_msg = ""
    session_t0 = time.time()

    # Filter resume skips first (serial, cheap)
    todo: list[tuple[int, str, str, dict, str]] = []
    for i, (label, name, rin, actor_id) in enumerate(planned_full, start=1):
        corpus_dir = ROOT / "data" / "raw" / (
            "pilot" if args.mode == "pilot" else f"corpus_{label}"
        )
        if args.resume and args.mode == "full":
            existing = already_done(corpus_dir, name)
            if existing:
                print(f"  SKIP (resume): {name} → {existing.name}", flush=True)
                log_rows.append(
                    {
                        "corpus": label,
                        "run_name": name,
                        "status": "skipped_resume",
                        "path": str(existing),
                    }
                )
                continue
        todo.append((i, label, name, rin, actor_id))

    print(f"Runs to execute (after resume skips): {len(todo)}", flush=True)

    def _worker(job: tuple[int, str, str, dict, str]) -> dict:
        i, label, name, rin, actor_id = job
        try:
            return execute_one_run(
                label=label,
                name=name,
                rin=rin,
                actor_id=actor_id,
                mode=args.mode,
                stamp=stamp,
                timeout=args.timeout,
                use_cli=use_cli,
                client=client,
                registry=registry,
                print_lock=print_lock,
            )
        except BudgetExhaustedError as exc:
            return {
                "corpus": label,
                "run_name": name,
                "status": "budget_exhausted",
                "error": str(exc)[:2000],
            }
        except Exception as exc:  # noqa: BLE001
            err = str(exc)
            if looks_like_budget_error(err):
                status = "budget_exhausted"
            elif looks_like_concurrency_error(err):
                status = "concurrency_limit"
            else:
                status = "failed"
            return {
                "corpus": label,
                "run_name": name,
                "status": status,
                "error": err[:2000],
            }

    # Parallel fan-out
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_worker, job): job for job in todo}
        for fut in as_completed(futures):
            row = fut.result()
            with log_lock:
                log_rows.append(row)
                if row.get("status") == "ok":
                    total_raw += int(row.get("n_raw") or 0)
                    total_new += int(row.get("n_new") or 0)
                    total_dup += int(row.get("n_dup") or 0)
                if row.get("status") == "budget_exhausted":
                    budget_hit = True
                    budget_msg = row.get("error") or "budget"
                    print("\n!!! APIFY BALANCE / PLAN LIMIT HIT !!!", flush=True)
                    print(str(budget_msg)[:1000], flush=True)
                    # cancel pending
                    for f in futures:
                        f.cancel()
                    break
            with print_lock:
                print(
                    f"  Session totals: raw={total_raw} new={total_new} "
                    f"dup={total_dup} | registry={len(registry)} IDs | "
                    f"elapsed={time.time() - session_t0:.0f}s",
                    flush=True,
                )

    registry.save()
    session_secs = time.time() - session_t0
    log_path = ROOT / "data" / "raw" / f"scrape_log_{args.mode}_{stamp}.json"
    ok_rows = [r for r in log_rows if r.get("status") == "ok"]
    avg_sec_run = (
        sum(r.get("seconds") or 0 for r in ok_rows) / len(ok_rows) if ok_rows else None
    )
    avg_sec_post = (
        sum(r.get("seconds") or 0 for r in ok_rows) / total_raw if total_raw else None
    )
    summary = {
        "stamp": stamp,
        "mode": args.mode,
        "workers": workers,
        "total_raw": total_raw,
        "total_new": total_new,
        "total_dup": total_dup,
        "registry_size": len(registry),
        "session_seconds": round(session_secs, 1),
        "avg_seconds_per_run": round(avg_sec_run, 1) if avg_sec_run else None,
        "avg_seconds_per_post_wall": round(avg_sec_post, 2) if avg_sec_post else None,
        "budget_exhausted": budget_hit,
        "budget_message": budget_msg if budget_hit else None,
        "runs": log_rows,
    }
    log_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nScrape log → {log_path}", flush=True)
    print(
        f"Done: {len(ok_rows)} ok runs | raw={total_raw} new={total_new} "
        f"dup={total_dup} | wall={session_secs:.0f}s | unique registry={len(registry)}",
        flush=True,
    )
    if avg_sec_post:
        print(
            f"Avg wall-time per raw post (sum of run times / posts): {avg_sec_post:.1f}s",
            flush=True,
        )
        if workers > 1 and session_secs > 0 and total_raw:
            effective = session_secs / total_raw
            print(
                f"Effective throughput with {workers} workers: {effective:.1f}s/post "
                f"({total_raw / (session_secs / 60):.1f} posts/min)",
                flush=True,
            )

    if budget_hit:
        print(
            "\n"
            + "=" * 60
            + "\nAPIFY CREDITS EXHAUSTED (or plan limit reached).\n"
            f"Partial this session: raw={total_raw} new={total_new}\n"
            "Log in with a new funded account, then resume:\n"
            "  python scripts/01_scrape_apify.py --mode full --corpus both "
            "--engine trudax --searches-only --resume --workers 3\n"
            + "=" * 60,
            flush=True,
        )
        return BUDGET_EXIT_CODE

    return 0 if ok_rows or any(r.get("status") == "skipped_resume" for r in log_rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
