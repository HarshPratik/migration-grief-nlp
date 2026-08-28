"""Phase 1 scrape via Arctic Shift (free Reddit archive, no Apify credits).

Walks every post in the configured subreddits from filters.date_from to now,
keeps self-text posts that already look like they will pass Phase 1 filters,
and writes trudax-compatible dumps under data/raw/corpus_{a,b}/.

Resume state: data/raw/arctic_shift_state.json

  python scripts/31_scrape_arctic_shift.py
  python scripts/31_scrape_arctic_shift.py --corpus b --sleep 0.35
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from seen_ids import SeenIdRegistry  # noqa: E402

FILTERS_PATH = ROOT / "config" / "filters.yaml"
RAW = ROOT / "data" / "raw"
STATE_PATH = RAW / "arctic_shift_state.json"
BASE = "https://arctic-shift.photon-reddit.com"
FIELDS = (
    "id,title,selftext,created_utc,subreddit,author,score,"
    "link_flair_text,url,over_18,num_comments,post_hint"
)
EMPTY_BODY = {"", "[removed]", "[deleted]"}
UA = "migration-grief-nlp/1.0 (academic research; posts-only archive replay)"


def load_filters() -> dict:
    with FILTERS_PATH.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def word_count(text: str) -> int:
    if not text:
        return 0
    return len(re.findall(r"\b\w+\b", text))


def exclude_re(keywords: list[str]) -> re.Pattern | None:
    alts = [
        re.escape(k.lower().strip()).replace(r"\ ", r"\s+")
        for k in keywords
        if k and str(k).strip()
    ]
    if not alts:
        return None
    return re.compile(
        r"(?<![\w-])(?:" + "|".join(alts) + r")(?![\w-])",
        flags=re.IGNORECASE,
    )


def iso_utc(ts: int | float) -> str:
    return datetime.fromtimestamp(int(ts), tz=timezone.utc).isoformat()


def _month_start(dt: datetime) -> datetime:
    return datetime(dt.year, dt.month, 1, tzinfo=timezone.utc)


def _add_month(dt: datetime) -> datetime:
    if dt.month == 12:
        return datetime(dt.year + 1, 1, 1, tzinfo=timezone.utc)
    return datetime(dt.year, dt.month + 1, 1, tzinfo=timezone.utc)


def month_windows(start: datetime, end: datetime) -> list[tuple[datetime, datetime]]:
    """Half-open [start, end) windows aligned to calendar months."""
    windows: list[tuple[datetime, datetime]] = []
    cursor = start
    guard = 0
    while cursor < end and guard < 120:
        guard += 1
        nxt = _add_month(_month_start(cursor))
        windows.append((cursor, min(nxt, end)))
        cursor = nxt
    return windows


def to_item(raw: dict, corpus: str) -> dict | None:
    body = (raw.get("selftext") or "").strip()
    if body.lower() in EMPTY_BODY:
        return None
    title = (raw.get("title") or "").strip()
    pid = str(raw.get("id") or "").removeprefix("t3_").strip()
    if not pid:
        return None
    sub = str(raw.get("subreddit") or "").replace("r/", "").strip()
    created = raw.get("created_utc")
    try:
        created_i = int(created)
    except (TypeError, ValueError):
        return None
    url = raw.get("url") or ""
    if not url and sub:
        url = f"https://www.reddit.com/r/{sub}/comments/{pid}/"
    return {
        "dataType": "post",
        "id": pid,
        "parsedId": pid,
        "title": title,
        "body": body,
        "selftext": body,
        "parsedCommunityName": sub,
        "communityName": sub,
        "subreddit": sub,
        "createdAt": iso_utc(created_i),
        "created_utc": created_i,
        "url": url,
        "flair": raw.get("link_flair_text") or "",
        "score": raw.get("score"),
        "username": raw.get("author") or "",
        "author": raw.get("author") or "",
        "over_18": raw.get("over_18"),
        "num_comments": raw.get("num_comments"),
        "_corpus_hint": corpus,
        "_source": "arctic_shift",
    }


def load_state() -> dict:
    if STATE_PATH.is_file():
        try:
            return json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {"subreddits": {}, "totals": {"a": 0, "b": 0}}


def save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")


def write_dump(corpus: str, sub: str, items: list[dict], n_dup: int) -> Path | None:
    if not items:
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    folder = RAW / f"corpus_{corpus}"
    folder.mkdir(parents=True, exist_ok=True)
    name = f"arctic_{sub}_{stamp}.json"
    path = folder / name
    meta = {
        "corpus": corpus,
        "run_name": f"arctic_{sub}",
        "source": "arctic_shift",
        "subreddit": sub,
        "n_items": len(items),
        "n_dup_skipped_before_write": n_dup,
    }
    path.write_text(
        json.dumps({"meta": meta, "items": items}, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return path


class ArcticClient:
    def __init__(self, sleep_s: float, timeout: float) -> None:
        self.sleep_s = sleep_s
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": UA, "Accept": "application/json"})

    def search(
        self,
        subreddit: str,
        after: int,
        before: int,
        limit: int | str,
    ) -> list[dict]:
        params = {
            "subreddit": subreddit,
            "after": after,
            "before": before,
            "sort": "asc",
            "limit": limit,
            "fields": FIELDS,
        }
        url = f"{BASE}/api/posts/search"
        last_err: Exception | None = None
        for attempt in range(8):
            try:
                r = self.session.get(url, params=params, timeout=self.timeout)
            except requests.RequestException as exc:
                last_err = exc
                time.sleep(min(30, 2 ** attempt))
                continue
            if r.status_code == 429:
                wait = r.headers.get("X-RateLimit-Reset") or "8"
                try:
                    wait_s = max(2.0, float(wait))
                except ValueError:
                    wait_s = 8.0
                print(f"    429 rate limit — sleep {wait_s:.0f}s", flush=True)
                time.sleep(wait_s)
                continue
            if r.status_code in (500, 502, 503, 504):
                time.sleep(min(30, 2 ** attempt))
                continue
            if r.status_code in (400, 413, 422):
                try:
                    err = (r.json() or {}).get("error") or r.text
                except Exception:
                    err = r.text
                raise RuntimeError(f"arctic {r.status_code}: {err}")
            r.raise_for_status()
            payload = r.json()
            if isinstance(payload, dict) and payload.get("error"):
                raise RuntimeError(str(payload["error"]))
            data = payload.get("data") if isinstance(payload, dict) else payload
            if not isinstance(data, list):
                return []
            time.sleep(self.sleep_s)
            return [x for x in data if isinstance(x, dict)]
        raise RuntimeError(f"arctic search failed after retries: {last_err}")


def scrape_window(
    client: ArcticClient,
    sub: str,
    after_ts: int,
    before_ts: int,
) -> list[dict]:
    """Paginate one [after, before) window. after is inclusive-ish; we dedupe."""
    out: list[dict] = []
    cursor = after_ts
    limit: int | str = 100
    idle_rounds = 0
    while cursor < before_ts:
        try:
            page = client.search(sub, after=cursor, before=before_ts, limit=limit)
        except RuntimeError as exc:
            msg = str(exc).lower()
            hard = any(s in msg for s in ("timed out", "400", "422", "413"))
            if hard and limit == 100:
                limit = 25
                print(f"    shrink limit=25 ({exc})", flush=True)
                continue
            if hard and (before_ts - cursor) > 3600:
                mid = cursor + (before_ts - cursor) // 2
                print(f"    bisect window {cursor}->{mid}->{before_ts} ({exc})", flush=True)
                left = scrape_window(client, sub, cursor, mid)
                right = scrape_window(client, sub, mid, before_ts)
                return out + left + right
            raise
        if not page:
            break
        # progress cursor
        last_ts = None
        for it in page:
            try:
                last_ts = int(it.get("created_utc"))
            except (TypeError, ValueError):
                continue
            out.append(it)
        if last_ts is None:
            break
        if last_ts <= cursor:
            cursor = last_ts + 1
            idle_rounds += 1
        else:
            cursor = last_ts
            idle_rounds = 0
        if idle_rounds >= 5:
            break
    return out


def keep_item(
    item: dict,
    corpus: str,
    min_wc: int,
    date_from_ts: int,
    a_exclude: re.Pattern | None,
) -> bool:
    created = int(item["created_utc"])
    if created < date_from_ts:
        return False
    if item.get("over_18") in (True, "true", "True", 1, "1"):
        return False
    # Phase 1 brief: the >100-word rule applies to the main body only, not
    # title+body. The blacklist still scans title+body (a visa question can
    # live entirely in the title).
    if word_count(item.get("body") or "") <= min_wc:
        return False
    text = f"{item.get('title') or ''}\n{item.get('body') or ''}".strip()
    if corpus == "a" and a_exclude is not None and a_exclude.search(text or ""):
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Scrape Reddit via Arctic Shift")
    parser.add_argument("--corpus", choices=("a", "b", "both"), default="both")
    parser.add_argument("--sleep", type=float, default=0.3)
    parser.add_argument("--timeout", type=float, default=90)
    parser.add_argument("--dump-size", type=int, default=400)
    parser.add_argument(
        "--target",
        type=int,
        default=0,
        help="Stop a SUBREDDIT after this many kept posts (0 = walk the full date window). "
        "A corpus-level cap is deliberately not offered: it silently starved later "
        "subreddits (r/bereavement collected 0 final posts under the old behaviour).",
    )
    parser.add_argument(
        "--no-target-cap",
        action="store_true",
        help="Collect every qualifying post in the date window (ignore --target)",
    )
    args = parser.parse_args()

    filters = load_filters()
    date_from = datetime.fromisoformat(str(filters["date_from"])).replace(
        tzinfo=timezone.utc
    )
    now = datetime.now(timezone.utc)
    date_from_ts = int(date_from.timestamp())
    min_wc = int(filters["min_word_count"])
    a_exclude = exclude_re(filters.get("exclude_keywords_corpus_a") or [])
    target = None if args.no_target_cap or int(args.target) <= 0 else int(args.target)

    jobs: list[tuple[str, str]] = []
    # Interleave A/B so a crash still grows both corpora.
    a_subs = [str(s) for s in filters["subreddits"]["corpus_a"]]
    b_subs = [str(s) for s in filters["subreddits"]["corpus_b"]]
    if args.corpus == "a":
        jobs = [("a", s) for s in a_subs]
    elif args.corpus == "b":
        jobs = [("b", s) for s in b_subs]
    else:
        i = j = 0
        while i < len(a_subs) or j < len(b_subs):
            if i < len(a_subs):
                jobs.append(("a", a_subs[i]))
                i += 1
            if j < len(b_subs):
                jobs.append(("b", b_subs[j]))
                j += 1

    registry = SeenIdRegistry(RAW / "seen_post_ids.json")
    state = load_state()
    state.setdefault("subreddits", {})
    state.setdefault("totals", {"a": 0, "b": 0})
    client = ArcticClient(sleep_s=args.sleep, timeout=args.timeout)
    windows = month_windows(date_from, now)

    print(
        f"Arctic Shift scrape  date_from={date_from.date()}  now={now.date()}  "
        f"windows={len(windows)}  seen_ids={len(registry)}  "
        f"target={'all' if target is None else target}/corpus",
        flush=True,
    )

    for corpus, sub in jobs:
        key = f"{corpus}:{sub}"
        sub_state = state["subreddits"].setdefault(
            key,
            {"done_until": date_from_ts, "fetched": 0, "kept": 0, "done": False},
        )
        if sub_state.get("done"):
            print(f"skip r/{sub} — already done", flush=True)
            continue

        print(f"\n=== corpus {corpus}  r/{sub} ===", flush=True)
        batch: list[dict] = []
        batch_dups = 0

        def flush_batch() -> None:
            nonlocal batch, batch_dups
            path = write_dump(corpus, sub, batch, batch_dups)
            if path:
                print(
                    f"  wrote {len(batch)} kept → {path.name}  "
                    f"corpus_{corpus}_kept={state['totals'][corpus]}",
                    flush=True,
                )
            registry.save()
            save_state(state)
            batch = []
            batch_dups = 0

        resume_from = int(sub_state.get("done_until") or date_from_ts)
        for w_start, w_end in windows:
            if target is not None and int(sub_state.get("kept") or 0) >= target:
                print(f"  r/{sub} hit per-subreddit target {target}", flush=True)
                break
            w0, w1 = int(w_start.timestamp()), int(w_end.timestamp())
            if w1 <= resume_from:
                continue
            after = max(w0, resume_from)
            label = w_start.strftime("%Y-%m")
            print(f"  window {label} after={after} before={w1}", flush=True)
            try:
                raw_posts = scrape_window(client, sub, after, w1)
            except Exception as exc:
                print(f"  ERROR r/{sub} {label}: {exc}", flush=True)
                flush_batch()
                raise
            mapped = []
            for raw in raw_posts:
                item = to_item(raw, corpus)
                if item is None:
                    continue
                if keep_item(item, corpus, min_wc, date_from_ts, a_exclude):
                    mapped.append(item)
            new_items, n_new, n_dup = registry.filter_new(mapped)
            sub_state["fetched"] = int(sub_state.get("fetched") or 0) + len(raw_posts)
            sub_state["kept"] = int(sub_state.get("kept") or 0) + n_new
            state["totals"][corpus] = int(state["totals"].get(corpus, 0)) + n_new
            batch.extend(new_items)
            batch_dups += n_dup
            sub_state["done_until"] = w1
            print(
                f"    fetched={len(raw_posts)} pass_filter={len(mapped)} "
                f"new={n_new} dup={n_dup} running_kept[{corpus}]="
                f"{state['totals'][corpus]}",
                flush=True,
            )
            if len(batch) >= args.dump_size:
                flush_batch()

        flush_batch()
        if target is None or int(sub_state.get("kept") or 0) < target:
            # only mark done if we walked every window
            last_end = int(windows[-1][1].timestamp()) if windows else resume_from
            if int(sub_state.get("done_until") or 0) >= last_end:
                sub_state["done"] = True
        save_state(state)
        print(
            f"  done r/{sub} fetched={sub_state['fetched']} kept={sub_state['kept']} "
            f"done={sub_state.get('done')}",
            flush=True,
        )

    registry.save()
    save_state(state)
    print(
        "\nTOTALS kept this campaign (plus prior campaign in state.totals): "
        f"A={state['totals'].get('a', 0)} B={state['totals'].get('b', 0)}",
        flush=True,
    )
    print("Next: python scripts/03_filter_and_normalize.py", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
