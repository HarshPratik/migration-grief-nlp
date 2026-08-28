"""
Phase 1 — Normalize raw JSON dumps, filter, dedupe, sample → interim corpora.

Sources are mixed (Arctic Shift, earlier Apify/trudax/pilot dumps). This script
does not call Apify.

  python scripts/03_filter_and_normalize.py
  python scripts/03_filter_and_normalize.py --input-glob "data/raw/**/*.json"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
FILTERS_PATH = ROOT / "config" / "filters.yaml"


def load_filters() -> dict:
    with FILTERS_PATH.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def word_count(text: str) -> int:
    if not text or not isinstance(text, str):
        return 0
    return len(re.findall(r"\b\w+\b", text))


def _should_skip_path(path: Path) -> bool:
    name = path.name.lower()
    if name.startswith(
        (
            "scrape_log",
            "apify_",
            "fire_and",
            "full_scrape",
            "home_batch",
            "fresh_batch",
            "cloud_download",
            "seen_post",
            "arctic_shift_state",
        )
    ):
        return True
    if "manifest" in name or name.endswith("_report.json") or name.endswith("_list.json"):
        return True
    if name.startswith("."):
        return True
    return False


def load_raw_items(paths: list[Path]) -> list[dict]:
    items: list[dict] = []
    for path in paths:
        if _should_skip_path(path):
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  skip {path}: {exc}")
            continue
        # Support wrapped {meta, items} and bare list / NDJSON-ish list
        if isinstance(data, dict) and "items" in data:
            batch = data["items"]
            corpus_hint = (data.get("meta") or {}).get("corpus")
        elif isinstance(data, list):
            batch = data
            corpus_hint = None
        else:
            print(f"  skip {path.name}: unrecognized JSON shape")
            continue
        if not isinstance(batch, list):
            print(f"  skip {path.name}: items not a list")
            continue
        # Path-based corpus hint for cloud dumps without meta.corpus
        if not corpus_hint:
            p = str(path).replace("\\", "/").lower()
            if "/corpus_a/" in p:
                corpus_hint = "a"
            elif "/corpus_b/" in p:
                corpus_hint = "b"
        for it in batch:
            if not isinstance(it, dict):
                continue
            if corpus_hint and "corpus" not in it:
                it = {**it, "_corpus_hint": corpus_hint}
            it["_source_file"] = str(path)
            items.append(it)
        print(f"  loaded {len(batch)} from {path.relative_to(ROOT) if ROOT in path.parents else path}")
    return items


def infer_corpus(row: dict, filters: dict) -> str | None:
    if row.get("_corpus_hint") in ("a", "b", "A", "B"):
        return str(row["_corpus_hint"]).lower()
    sub = (
        row.get("parsedCommunityName")
        or row.get("communityName")
        or row.get("subreddit")
        or row.get("category")
        or ""
    )
    sub = str(sub).replace("r/", "").strip()
    sub_l = sub.lower()
    for label, subs in filters["subreddits"].items():
        # label is corpus_a / corpus_b — match case-insensitively
        for s in subs:
            if sub_l == str(s).lower():
                return "a" if label.endswith("a") else "b"
    return None


def parse_created(row: dict) -> datetime | None:
    """Parse mixed Reddit timestamp formats. None means unparsable (fail-closed)."""
    utc = row.get("created_utc")
    if utc is not None and utc != "":
        try:
            ts = float(utc)
            if ts > 1e12:
                ts /= 1000.0
            return datetime.fromtimestamp(ts, tz=timezone.utc)
        except (TypeError, ValueError, OSError, OverflowError):
            pass
    created = row.get("createdAt") or row.get("created_at") or row.get("created")
    if created is None or created == "":
        return None
    if isinstance(created, datetime):
        dt = created
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    if isinstance(created, (int, float)):
        try:
            ts = float(created)
            if ts > 1e12:
                ts /= 1000.0
            return datetime.fromtimestamp(ts, tz=timezone.utc)
        except (TypeError, ValueError, OSError, OverflowError):
            return None
    s = str(created).strip()
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def normalize_item(row: dict, filters: dict) -> dict | None:
    """Normalize Arctic Shift, trudax, and labrat011 Reddit schemas."""
    data_type = row.get("dataType") or row.get("type")
    if data_type and str(data_type).lower() not in ("post",):
        return None

    title = (row.get("title") or "").strip()
    # labrat: selftext | trudax/arctic: body
    body = (row.get("selftext") or row.get("body") or "").strip()
    # Strip Reddit "submitted by /u/... [link] [comments]" footer from RSS-style scrapes
    body = re.sub(
        r"(?:&#32;\s*)*submitted by.*?(?:\[link\]|\[comments\]).*$",
        "",
        body,
        flags=re.I | re.S,
    ).strip()
    body = re.sub(r"&#\d+;", " ", body)
    body = re.sub(r"\s+", " ", body).strip()
    if not body:
        return None
    if body.lower() in ("[removed]", "[deleted]"):
        return None

    post_id = row.get("parsedId") or row.get("id")
    if not post_id:
        return None
    post_id = str(post_id).removeprefix("t3_")

    sub = (
        row.get("parsedCommunityName")
        or row.get("communityName")
        or row.get("subreddit")
        or row.get("category")
        or ""
    )
    sub = str(sub).replace("r/", "").strip()

    text = f"{title}\n{body}".strip() if title else body
    # attach sub for corpus inference
    row = {**row, "parsedCommunityName": sub}
    corpus = infer_corpus(row, filters)
    if corpus is None:
        return None

    created = parse_created(row)
    flair = (
        row.get("flair")
        or row.get("linkFlairText")
        or row.get("link_flair_text")
        or ""
    )
    if isinstance(flair, dict):
        flair = flair.get("text") or ""

    score = row.get("upVotes")
    if score is None:
        score = row.get("score")

    return {
        "post_id": post_id,
        "corpus": corpus,
        "subreddit": sub,
        "title": title,
        "body": body,
        "text": text,
        "created_at": created,
        "url": row.get("url") or "",
        "flair": str(flair) if flair else "",
        "score": score,
        "word_count": word_count(text),
        "body_word_count": word_count(body),
        "username": row.get("username") or row.get("author") or "",
        "_source_file": row.get("_source_file", ""),
    }


def flair_matches(flair: str, patterns: list[str]) -> bool:
    if not flair:
        return False
    f = flair.lower()
    return any(p.lower() in f for p in patterns)


def _exclude_pattern(keywords: list[str]) -> re.Pattern | None:
    """Compile the Corpus A blacklist as whole-word alternatives.

    Substring matching (`k in text`) was catastrophically over-broad here:
    "ead" matched already/read/head/instead/ahead/dead/spread and "irs" matched
    first/thirsty, so the blacklist dropped 1,151 of 1,480 eligible Corpus A
    posts instead of the intended 454. Measured false positives: ead 829,
    irs 349, tax 37, visa 24 (inside "advisable"), petition 20 ("competition").
    See scripts/07_audit_corpus_recovery.py for the full reconciliation.

    Hyphenated keys such as "i-485" and "h-1b" are matched with lookarounds
    rather than \\b, because \\b does not behave as intended next to a hyphen.
    """
    if not keywords:
        return None
    alternatives = [
        re.escape(k.lower().strip()).replace(r"\ ", r"\s+")
        for k in keywords
        if k and k.strip()
    ]
    if not alternatives:
        return None
    return re.compile(
        r"(?<![\w-])(?:" + "|".join(alternatives) + r")(?![\w-])",
        flags=re.IGNORECASE,
    )


def contains_exclude(text: str, keywords: list[str]) -> bool:
    pattern = _exclude_pattern(keywords)
    if pattern is None:
        return False
    return bool(pattern.search(text or ""))


def _rng(seed: int, *parts: str) -> np.random.Generator:
    payload = f"{seed}|{'|'.join(parts)}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def proportional_quotas(available: dict[str, int], target: int) -> dict[str, int]:
    """Largest-remainder allocation, never assigning more than available."""
    avail = {k: int(v) for k, v in available.items()}
    total = sum(avail.values())
    if total <= target:
        return avail
    quotas = {k: 0 for k in avail}
    assigned = 0
    remainders: list[tuple[float, str]] = []
    for k, n in avail.items():
        raw = (n * target) / total
        q = min(n, int(raw))
        quotas[k] = q
        assigned += q
        remainders.append((raw - int(raw), k))
    leftover = target - assigned
    for _, k in sorted(remainders, reverse=True):
        if leftover <= 0:
            break
        if quotas[k] < avail[k]:
            quotas[k] += 1
            leftover -= 1
    if leftover > 0:
        for k, n in sorted(avail.items(), key=lambda kv: kv[1], reverse=True):
            take = min(n - quotas[k], leftover)
            quotas[k] += take
            leftover -= take
            if leftover <= 0:
                break
    return quotas


def scarce_first_quotas(available: dict[str, int], target: int) -> dict[str, int]:
    """Take every post from the smallest subreddits, fill the rest from larger ones.

    Used for Corpus B so r/bereavement is not crowded out by r/GriefSupport.
    """
    quotas = {k: 0 for k in available}
    remaining = int(target)
    for k, n in sorted(available.items(), key=lambda kv: (kv[1], kv[0])):
        take = min(int(n), remaining)
        quotas[k] = take
        remaining -= take
        if remaining <= 0:
            break
    return quotas


def sample_flair_first(group: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Take preferred-flair posts first; fill any remainder uniformly at random."""
    if n <= 0 or group.empty:
        return group.iloc[0:0]
    if len(group) <= n:
        return group
    pref = group[group["flair_preferred"]]
    rest = group[~group["flair_preferred"]]
    parts: list[pd.DataFrame] = []
    if len(pref) >= n:
        idx = rng.choice(pref.index.to_numpy(), size=n, replace=False)
        return pref.loc[idx]
    parts.append(pref)
    need = n - len(pref)
    if need > 0:
        if len(rest) < need:
            parts.append(rest)
        else:
            idx = rng.choice(rest.index.to_numpy(), size=need, replace=False)
            parts.append(rest.loc[idx])
    return pd.concat(parts)


def apply_filters(df: pd.DataFrame, filters: dict) -> tuple[pd.DataFrame, dict]:
    stats: dict = {"start": len(df)}
    date_from = pd.Timestamp(filters["date_from"], tz="UTC")
    date_to = (
        pd.Timestamp(filters["date_to"], tz="UTC") if filters.get("date_to") else None
    )
    min_wc = int(filters["min_word_count"])
    target = int(filters.get("target_posts_per_corpus", 10000))
    seed = int(filters.get("sampling_seed", 20260827))

    df = df.copy()
    if "body_word_count" not in df.columns:
        df["body_word_count"] = df["body"].map(word_count)
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")

    # Dates — fail-closed: unparsable timestamps are dropped, not kept.
    before = len(df)
    in_window = df["created_at"].notna() & (df["created_at"] >= date_from)
    if date_to is not None:
        in_window &= df["created_at"] < date_to
    stats["dropped_unparsable_date"] = int(df["created_at"].isna().sum())
    df = df[in_window]
    stats["after_date"] = len(df)
    stats["dropped_date"] = before - len(df)
    stats["date_from"] = str(date_from)
    stats["date_to"] = str(date_to) if date_to is not None else None
    stats["date_policy"] = "fail_closed"

    # Phase 1 brief: keep posts whose MAIN BODY is longer than 100 words.
    before = len(df)
    df = df[df["body_word_count"] > min_wc]
    stats["after_word_count"] = len(df)
    stats["dropped_word_count"] = before - len(df)
    stats["word_count_field"] = "body_word_count"

    include_flairs = filters.get("include_flairs") or []
    df["flair_preferred"] = df["flair"].apply(lambda x: flair_matches(str(x), include_flairs))
    df["flair_missing"] = df["flair"].fillna("").eq("")

    # Corpus A exclude keywords (title + body)
    exclude = filters.get("exclude_keywords_corpus_a") or []
    before = len(df)
    exclude_re = _exclude_pattern(exclude)
    mask_a_drop = (df["corpus"] == "a") & df["text"].apply(
        lambda t: bool(exclude_re.search(t or "")) if exclude_re is not None else False
    )
    df = df[~mask_a_drop]
    stats["after_exclude_keywords"] = len(df)
    stats["dropped_exclude_keywords"] = before - len(df)

    # Corpus A procedural flairs (visa / citizenship / red tape / tax / financial)
    exclude_flairs = filters.get("exclude_flairs_corpus_a") or []
    before = len(df)
    mask_flair_drop = (df["corpus"] == "a") & df["flair"].apply(
        lambda x: flair_matches(str(x), exclude_flairs)
    )
    df = df[~mask_flair_drop]
    stats["after_exclude_flairs"] = len(df)
    stats["dropped_exclude_flairs"] = before - len(df)

    # Dedupe: keep the longest body when IDs collide
    before = len(df)
    df = df.sort_values(["body_word_count"], ascending=False)
    df = df.drop_duplicates(subset=["post_id"], keep="first")
    stats["after_dedupe"] = len(df)
    stats["dropped_dupes"] = before - len(df)

    pool_counts = {
        f"{c}/{s}": int(n)
        for (c, s), n in df.groupby(["corpus", "subreddit"]).size().items()
    }
    stats["pool_by_subreddit"] = pool_counts

    # Per-subreddit quotas, then flair-first seeded random sampling.
    # Corpus A: proportional to the qualifying pool.
    # Corpus B: scarce-first so r/bereavement is fully included.
    parts: list[pd.DataFrame] = []
    quotas_out: dict[str, int] = {}
    sampled_flair: dict[str, int] = {}
    for corpus, group in df.groupby("corpus"):
        available = group.groupby("subreddit").size().astype(int).to_dict()
        if str(corpus) == "a":
            quotas = proportional_quotas(available, target)
            method = "proportional_to_pool"
        else:
            quotas = scarce_first_quotas(available, target)
            method = "scarce_first"
        stats[f"corpus_{corpus}_quota_method"] = method
        sampled_parts: list[pd.DataFrame] = []
        for sub, n in quotas.items():
            sub_g = group[group["subreddit"] == sub]
            rng = _rng(seed, str(corpus), str(sub))
            taken = sample_flair_first(sub_g, int(n), rng)
            sampled_parts.append(taken)
            key = f"{corpus}/{sub}"
            quotas_out[key] = int(n)
            sampled_flair[key] = int(taken["flair_preferred"].sum()) if len(taken) else 0
        g = pd.concat(sampled_parts, ignore_index=True) if sampled_parts else group.iloc[0:0]
        parts.append(g)
        stats[f"corpus_{corpus}_final"] = len(g)
        stats[f"corpus_{corpus}_flair_preferred"] = int(g["flair_preferred"].sum()) if len(g) else 0
    out = pd.concat(parts, ignore_index=True) if parts else df.iloc[0:0]
    stats["quotas"] = quotas_out
    stats["sampled_flair_preferred"] = sampled_flair
    stats["sampling_seed"] = seed
    stats["sampling"] = "flair_first_then_random_within_subreddit_quota"
    stats["end"] = len(out)
    return out, stats


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-glob",
        default="data/raw/**/*.json",
        help="Glob relative to repo root for raw JSON dumps",
    )
    args = parser.parse_args()

    filters = load_filters()
    paths = sorted(ROOT.glob(args.input_glob))
    # Also catch pilot files one level down
    if not paths:
        paths = sorted((ROOT / "data" / "raw").rglob("*.json"))

    print(f"Found {len(paths)} JSON files")
    raw = load_raw_items(paths)
    print(f"Total raw items: {len(raw)}")

    rows = []
    skipped = Counter()
    for it in raw:
        norm = normalize_item(it, filters)
        if norm is None:
            dt = it.get("dataType", "unknown")
            skipped[dt] += 1
            continue
        rows.append(norm)

    print(f"Normalized posts: {len(rows)} | skipped by type/empty: {dict(skipped)}")
    if not rows:
        print("No rows to filter. Run a scrape first.")
        return 1

    df = pd.DataFrame(rows)
    filtered, stats = apply_filters(df, filters)
    print("\nFilter stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    interim = ROOT / "data" / "interim"
    interim.mkdir(parents=True, exist_ok=True)

    for corpus in ("a", "b"):
        part = filtered[filtered["corpus"] == corpus].copy()
        out = interim / f"corpus_{corpus}.parquet"
        part.to_parquet(out, index=False)
        csv_out = interim / f"corpus_{corpus}.csv"
        part.to_csv(csv_out, index=False)
        print(f"Corpus {corpus.upper()}: {len(part)} rows → {out}")

    summary = {
        "filter_stats": stats,
        "by_subreddit": {},
    }
    if len(filtered):
        summary["by_subreddit"] = {
            f"{c}/{s}": int(n)
            for (c, s), n in filtered.groupby(["corpus", "subreddit"]).size().items()
        }
    (interim / "filter_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
