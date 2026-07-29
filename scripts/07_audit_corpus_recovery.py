"""Phase 2 - reconcile raw dumps against data/processed and explain every drop.

Motivation: 1,480 unique Corpus A posts over 100 words with non-empty bodies
exist in data/raw, but only 323 reach data/processed/corpus_a.parquet. The
documented filters (date >= 2023-07-01, word count > 100, visa/tax blacklist,
dedupe) do not account for that gap.

This script attributes every dropped post to exactly one reason, so the funnel
adds up and no loss is unexplained. It also reports what each blacklist keyword
would drop under substring matching (what scripts/03_filter_and_normalize.py:203
currently does) versus word-boundary matching (what the rule intends), because
that difference is where most of the gap lives.

Read-only: writes to outputs/ only, never to data/.

Example:
    D:\\research_paper\\.venv-analysis\\Scripts\\python.exe scripts/07_audit_corpus_recovery.py
    ... scripts/07_audit_corpus_recovery.py --corpus b --top-keywords 25
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analysis_io import banner, item, out_path, save_summary, save_table  # noqa: E402

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
FILTERS_PATH = ROOT / "config" / "filters.yaml"

# Filenames under data/raw that are logs/manifests, not post dumps.
SKIP_MARKERS = ("scrape_log", "apify_", "seen_post", "filter_summary", "phase1", "cloud_download")

SUBREDDITS = {
    "a": {"expats", "immigration", "studyabroad"},
    "b": {"griefsupport", "bereavement"},
}

# Order matters: each post is attributed to the FIRST reason that applies, so
# the funnel sums exactly to the input count.
DROP_ORDER = (
    "not_a_post",
    "empty_body",
    "no_post_id",
    "unknown_subreddit",
    "word_count",
    "date",
    "exclude_keywords",
    "duplicate",
)


def load_filters() -> dict:
    import yaml

    return yaml.safe_load(FILTERS_PATH.read_text(encoding="utf-8"))


def iter_raw_items(raw_dir: Path):
    """Yield (item, source_path) for every post-like record under data/raw."""
    for path in sorted(raw_dir.rglob("*.json")):
        if any(marker in path.name for marker in SKIP_MARKERS):
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            continue
        records = payload.get("items", payload) if isinstance(payload, dict) else payload
        if not isinstance(records, list):
            continue
        for record in records:
            if isinstance(record, dict):
                yield record, path


def normalize(record: dict) -> dict:
    """Minimal normalisation mirroring 03_filter_and_normalize.normalize_item."""
    body = str(record.get("selftext") or record.get("body") or "").strip()
    body = re.sub(
        r"(?:&#32;\s*)*submitted by.*?(?:\[link\]|\[comments\]).*$", "", body, flags=re.I | re.S
    ).strip()
    body = re.sub(r"&#\d+;", " ", body)
    body = re.sub(r"\s+", " ", body).strip()

    title = str(record.get("title") or "").strip()
    text = f"{title}\n{body}".strip() if title else body
    subreddit = str(
        record.get("parsedCommunityName")
        or record.get("communityName")
        or record.get("subreddit")
        or record.get("category")
        or ""
    ).replace("r/", "").strip()

    post_id = record.get("parsedId") or record.get("id") or ""
    return {
        "post_id": str(post_id).removeprefix("t3_"),
        "subreddit": subreddit,
        "subreddit_key": subreddit.lower(),
        "title": title,
        "body": body,
        "text": text,
        "word_count": len(text.split()),
        "created_at": record.get("createdAt") or record.get("created_at") or record.get("created"),
        "data_type": str(record.get("dataType") or record.get("type") or "post").lower(),
    }


def keyword_hits(text: str, keywords: list[str], *, word_boundary: bool) -> list[str]:
    lowered = text.lower()
    if not word_boundary:
        return [k for k in keywords if k.lower() in lowered]
    return [
        k for k in keywords
        if re.search(r"\b" + re.escape(k.lower()).replace(r"\ ", r"\s+") + r"\b", lowered)
    ]


def audit(corpus: str, filters: dict, *, word_boundary: bool) -> tuple[pd.DataFrame, dict]:
    """Attribute every raw record for one corpus to a keep/drop reason."""
    wanted = SUBREDDITS[corpus]
    date_from = pd.Timestamp(filters["date_from"], tz="UTC")
    min_words = int(filters["min_word_count"])
    keywords = list(filters.get("exclude_keywords_corpus_a") or []) if corpus == "a" else []

    best: dict[str, dict] = {}
    rows: list[dict] = []
    seen_rows = 0

    for record, path in iter_raw_items(RAW_DIR):
        norm = normalize(record)
        if norm["subreddit_key"] not in wanted:
            continue
        seen_rows += 1
        norm["_source_file"] = str(path.relative_to(ROOT))

        reason = None
        if norm["data_type"] not in ("post", ""):
            reason = "not_a_post"
        elif not norm["body"]:
            reason = "empty_body"
        elif not norm["post_id"]:
            reason = "no_post_id"

        if reason:
            rows.append({**norm, "drop_reason": reason})
            continue

        # Keep the longest version of each post, mirroring the dedupe rule.
        prior = best.get(norm["post_id"])
        if prior is None or len(norm["text"]) > len(prior["text"]):
            if prior is not None:
                rows.append({**prior, "drop_reason": "duplicate"})
            best[norm["post_id"]] = norm
        else:
            rows.append({**norm, "drop_reason": "duplicate"})

    # Now apply the content filters to the deduplicated survivors.
    kept = 0
    for norm in best.values():
        reason = None
        if norm["word_count"] <= min_words:
            reason = "word_count"
        else:
            created = pd.to_datetime(norm["created_at"], errors="coerce", utc=True)
            if pd.notna(created) and created < date_from:
                reason = "date"
            elif keywords:
                hits = keyword_hits(norm["text"], keywords, word_boundary=word_boundary)
                if hits:
                    reason = "exclude_keywords"
                    norm = {**norm, "keyword_hits": ";".join(hits)}
        if reason:
            rows.append({**norm, "drop_reason": reason})
        else:
            kept += 1
            rows.append({**norm, "drop_reason": ""})

    frame = pd.DataFrame(rows)
    counts = Counter(frame["drop_reason"])
    stats = {
        "corpus": corpus,
        "raw_rows_scanned": seen_rows,
        "unique_posts": len(best),
        "kept": kept,
        "dropped_by_reason": {r: counts.get(r, 0) for r in DROP_ORDER},
        "matching": "word_boundary" if word_boundary else "substring",
    }
    return frame, stats


def keyword_report(frame: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """Per-keyword substring vs word-boundary kill counts on eligible posts."""
    keywords = list(filters.get("exclude_keywords_corpus_a") or [])
    eligible = frame[frame["drop_reason"].isin(["", "exclude_keywords"])]
    rows = []
    for keyword in keywords:
        substring = sum(
            1 for t in eligible["text"] if keyword.lower() in str(t).lower()
        )
        boundary = sum(
            1 for t in eligible["text"]
            if re.search(r"\b" + re.escape(keyword.lower()).replace(r"\ ", r"\s+") + r"\b", str(t).lower())
        )
        rows.append({
            "keyword": keyword,
            "substring_hits": substring,
            "word_boundary_hits": boundary,
            "false_positives": substring - boundary,
            "false_positive_rate": round((substring - boundary) / substring, 4) if substring else 0.0,
        })
    return pd.DataFrame(rows).sort_values("false_positives", ascending=False)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reconcile raw dumps against data/processed and explain every dropped post"
    )
    parser.add_argument("--corpus", choices=["a", "b", "both"], default="both")
    parser.add_argument(
        "--substring",
        action="store_true",
        help="attribute drops using substring matching (reproduces the current 03 behaviour)",
    )
    parser.add_argument("--top-keywords", type=int, default=20)
    args = parser.parse_args()

    filters = load_filters()
    labels = ["a", "b"] if args.corpus == "both" else [args.corpus]
    word_boundary = not args.substring

    summary: dict = {"matching": "word_boundary" if word_boundary else "substring", "corpora": {}}

    for label in labels:
        banner(f"Audit corpus {label.upper()}")
        frame, stats = audit(label, filters, word_boundary=word_boundary)

        processed = PROCESSED_DIR / f"corpus_{label}.parquet"
        current = len(pd.read_parquet(processed)) if processed.exists() else 0
        stats["current_processed_rows"] = current
        stats["recoverable"] = stats["kept"] - current

        item(f"raw rows scanned      {stats['raw_rows_scanned']}")
        item(f"unique posts          {stats['unique_posts']}")
        for reason in DROP_ORDER:
            count = stats["dropped_by_reason"][reason]
            if count:
                item(f"dropped: {reason:<20} {count}", indent=2)
        item(f"would keep            {stats['kept']}")
        item(f"currently in parquet  {current}")
        item(f"RECOVERABLE           {stats['recoverable']:+d}")

        dropped = frame[frame["drop_reason"] != ""].copy()
        columns = [
            "post_id", "subreddit", "word_count", "created_at",
            "drop_reason", "keyword_hits", "_source_file",
        ]
        for column in columns:
            if column not in dropped.columns:
                dropped[column] = ""
        save_table(
            dropped[columns],
            out_path("tables", "p2", 7, "dropped_posts", label),
            parquet=False,
        )

        if label == "a":
            report = keyword_report(frame, filters)
            save_table(report, out_path("tables", "p2", 7, "keyword_false_positives"), parquet=False)
            banner("Blacklist keyword false positives (Corpus A)")
            item(f"{'keyword':<22}{'substr':>8}{'word':>8}{'false+':>8}")
            for _, row in report.head(args.top_keywords).iterrows():
                if row["substring_hits"]:
                    item(
                        f"{row['keyword']:<22}{row['substring_hits']:>8}"
                        f"{row['word_boundary_hits']:>8}{row['false_positives']:>8}"
                    )
            stats["keyword_false_positives_total"] = int(report["false_positives"].sum())

        summary["corpora"][label] = stats

    save_summary(summary, out_path("summaries", "p2", 7, "audit", ext="json"))
    banner("Done")
    item(f"tables  -> {out_path('tables', 'p2', 7, 'dropped_posts', 'a').parent}")
    item(f"summary -> {out_path('summaries', 'p2', 7, 'audit', ext='json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
