"""Shared scrape plan: build run list + detect completed local dumps."""

from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "scrape_runs.yaml"
FILTERS_PATH = ROOT / "config" / "filters.yaml"
RAW = ROOT / "data" / "raw"


def load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_trudax_search_runs(
    label: str,
    subreddits: list[str],
    queries: list[str],
    cfg: dict,
    limits: dict,
) -> list[tuple[str, str, dict]]:
    """Return (corpus, run_name, input) for trudax search-only runs."""
    runs: list[tuple[str, str, dict]] = []
    base = dict(cfg.get("trudax") or {})
    max_items = int(limits.get("maxItems", 80))
    max_posts = int(limits.get("maxPostCount", 80))
    batch_size = 5
    for sub in subreddits:
        for bi in range(0, len(queries), batch_size):
            batch = queries[bi : bi + batch_size]
            name = f"trudax_{sub}_search_batch{bi // batch_size}"
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


def planned_search_runs(
    corpus: str = "both",
    mode: str = "full",
) -> list[tuple[str, str, dict, str]]:
    """Full planned list: (corpus, run_name, input, actor_id)."""
    cfg = load_yaml(CONFIG_PATH)
    filters = load_yaml(FILTERS_PATH)
    limits = dict(cfg[mode])
    actor_id = cfg.get("primary_actor_id") or "trudax/reddit-scraper-lite"

    corpora = []
    if corpus in ("a", "both"):
        corpora.append(
            ("a", filters["subreddits"]["corpus_a"], cfg["search_queries"]["corpus_a"])
        )
    if corpus in ("b", "both"):
        corpora.append(
            ("b", filters["subreddits"]["corpus_b"], cfg["search_queries"]["corpus_b"])
        )

    planned: list[tuple[str, str, dict]] = []
    for label, subs, queries in corpora:
        planned.extend(build_trudax_search_runs(label, subs, queries, cfg, limits))
    planned = interleave_runs(planned)
    return [(lab, name, rin, actor_id) for lab, name, rin in planned]


def dump_item_count(path: Path) -> int:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return 0
    items = data.get("items") if isinstance(data, dict) else data
    return len(items) if isinstance(items, list) else 0


def completed_run_names(min_items: int = 10) -> dict[str, dict]:
    """
    Map run_name -> best local dump info.

    A run is completed only if some dump has >= min_items posts.
    Tiny dumps (timeouts with 1 item) are NOT completed → can re-queue.
    """
    done: dict[str, dict] = {}
    for corpus in ("a", "b"):
        d = RAW / f"corpus_{corpus}"
        if not d.is_dir():
            continue
        for path in d.glob("*.json"):
            # run_name is filename without trailing _TIMESTAMP.json
            # e.g. trudax_expats_search_batch0_new_year_20260722T134841Z.json
            m = re.match(r"^(trudax_.+)_(\d{8}T\d{6}Z)\.json$", path.name)
            if m:
                run_name = m.group(1)
            else:
                # partials / other names: skip as "completed plan names"
                continue
            n = dump_item_count(path)
            prev = done.get(run_name)
            if prev is None or n > prev["n_items"]:
                done[run_name] = {
                    "n_items": n,
                    "path": str(path),
                    "corpus": corpus,
                    "complete": n >= min_items,
                }
    return done


def unfinished_runs(
    corpus: str = "both",
    mode: str = "full",
    min_items: int = 10,
) -> tuple[list[tuple[str, str, dict, str]], list[dict]]:
    """
    Returns (todo, skipped_info).
    todo: runs to start on Apify
    skipped: completed with enough items
    """
    planned = planned_search_runs(corpus=corpus, mode=mode)
    status = completed_run_names(min_items=min_items)
    todo = []
    skipped = []
    for lab, name, rin, actor_id in planned:
        info = status.get(name)
        if info and info.get("complete"):
            skipped.append(
                {
                    "corpus": lab,
                    "run_name": name,
                    "n_items": info["n_items"],
                    "path": info["path"],
                    "reason": f"local dump >= {min_items} items",
                }
            )
            continue
        reason = None
        if info and not info.get("complete"):
            reason = f"local dump only {info['n_items']} items (retry)"
        todo.append((lab, name, rin, actor_id, reason))
    return todo, skipped
