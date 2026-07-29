"""Phase 3 - extract +/-N token context windows around every anchor occurrence.

This is the substrate for the emotion / ABSA stage. Nothing here scores
anything; it decides WHAT TEXT gets scored, which is the decision that most
constrains everything downstream.

Design notes, in the order they bite:

* WHICH TOKENS. Windows are cut from `tokens_ctx` (surface tokens, STOPWORDS
  RETAINED) and never from `tokens_lemma`. "+/-5 words" means five surface
  words; and an emotion model handed a stopword-stripped window has had its
  negators deleted, which inverts polarity silently.
* SENTENCE CLIPPING. A window spilling into the next sentence measures that
  sentence's emotion, not the anchor's. On by default; `--no-clip-sentence`
  turns it off for the robustness comparison.
* NEGATION RESCUE. Clipping can amputate a negator ("I do not | miss home").
  `--negation-rescue N` pulls the left edge back to a negator within N tokens,
  still inside the anchor's own sentence.
* COST. A token -> positions index is built once per document
  (`build_token_index`), so extraction is O(hits). Looping 71 anchors x ~350k
  surface tokens instead would be ~10^8 comparisons.
* ROBUSTNESS. Windows are emitted at EVERY configured size (primary plus
  `windows.robustness_sizes`) in one pass, so the "does the result survive at
  +/-3 and +/-10" check is possible later without re-running the pipeline.

EXHAUSTIVE DIGITAL CONCORDANCE. Every Digital-domain hit in Corpus A is
written out with an empty `manual_category` column for hand-coding
(interpersonal / information-seeking / logistics / other). At this count the
enumeration is COMPLETE, and that is a methodological strength rather than a
limitation: the domain's interpretation rests on the full population of hits,
not on a sample of them, so no sampling error enters the coding step and the
resulting proportions are exact. It is also the only defensible way to handle
the polysemous anchors (`connection`, `disconnect`, `signal`), whose raw
counts otherwise overstate what the domain measures.

Examples:
    D:\\research_paper\\.venv-analysis\\Scripts\\python.exe scripts/11_extract_windows.py
    ... scripts/11_extract_windows.py --corpus a --size 5
    ... scripts/11_extract_windows.py --no-clip-sentence --no-negation-rescue
    ... scripts/11_extract_windows.py --robustness-sizes 3 10 20 --no-dedupe
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analysis_config import load_analysis_config, load_anchors  # noqa: E402
from analysis_io import as_token_list, banner, item, out_path, save_summary, save_table  # noqa: E402
from text_windows import (  # noqa: E402
    NEGATORS,
    Window,
    build_token_index,
    dedupe_overlapping,
    extract_windows,
    find_anchor_positions,
    kwic_line,
)

EXHAUSTIVE_DOMAIN = "digital"
EXHAUSTIVE_CORPUS = "a"
MANUAL_CATEGORIES = ("interpersonal", "information-seeking", "logistics", "other")

WINDOW_COLUMNS = (
    "post_id", "corpus", "subreddit", "anchor", "domain", "polysemous", "window_size",
    "hit_index", "position", "sent_idx", "start", "end", "n_left", "n_right",
    "truncated_left", "truncated_right", "negation_rescued",
    "window_tokens", "window_text", "left_text", "right_text",
)


def sentence_index(cell: Any, n_tokens: int) -> list[int] | None:
    """`sent_idx` as ints, or None when it is absent or misaligned.

    Parquet hands list columns back as ndarray and `as_token_list` stringifies
    them, so the ints have to be restored explicitly. A length mismatch is
    returned as None rather than raised: the window extractor then treats the
    document as unsegmented instead of aborting the whole run.
    """
    values = as_token_list(cell)
    if not values or len(values) != n_tokens:
        return None
    try:
        return [int(float(v)) for v in values]
    except (TypeError, ValueError):
        return None


def window_record(
    window: Window,
    tokens: Sequence[str],
    row: Mapping[str, Any],
    *,
    domain: str,
    polysemous: bool,
    size: int,
    hit_index: int,
) -> dict[str, Any]:
    """Flatten one Window into the output schema. Never mutates its inputs."""
    span = [str(t) for t in tokens[window.start : window.end + 1]]
    return {
        "post_id": row["post_id"],
        "corpus": row["corpus"],
        "subreddit": row["subreddit"],
        "anchor": window.anchor,
        "domain": domain,
        "polysemous": polysemous,
        "window_size": size,
        "hit_index": hit_index,
        "position": window.position,
        "sent_idx": window.sent_idx,
        "start": window.start,
        "end": window.end,
        "n_left": window.n_left,
        "n_right": window.n_right,
        "truncated_left": window.truncated_left,
        "truncated_right": window.truncated_right,
        "negation_rescued": window.negation_rescued,
        "window_tokens": span,
        "window_text": " ".join(span),
        "left_text": " ".join(str(t) for t in tokens[window.start : window.position]),
        "right_text": " ".join(str(t) for t in tokens[window.position + 1 : window.end + 1]),
        "kwic": kwic_line(tokens, window),
    }


def extract_corpus(
    df: pd.DataFrame,
    label: str,
    *,
    terms: frozenset[str],
    domain_of: Mapping[str, str],
    polysemous_of: Mapping[str, bool],
    sizes: Sequence[int],
    primary: int,
    clip_at_sentence: bool,
    negation_rescue: int,
    max_per_anchor: int | None,
    dedupe: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    """Windows at every requested size for one corpus.

    Returns (records, exhaustive_records, n_docs_with_hits). The second list is
    the UNDEDUPED primary-size hits, which is what "exhaustive" has to mean:
    dedupe drops a second occurrence lying inside the first one's span, and an
    exhaustive concordance must show it.
    """
    records: list[dict[str, Any]] = []
    exhaustive: list[dict[str, Any]] = []
    docs_with_hits = 0

    for row in df.to_dict("records"):
        tokens = as_token_list(row["tokens_ctx"])
        if not tokens:
            continue
        sents = sentence_index(row.get("sent_idx"), len(tokens))
        index = build_token_index(tokens)
        positions = find_anchor_positions(tokens, terms, index=index)
        if not positions:
            continue
        docs_with_hits += 1

        for size in sizes:
            found = extract_windows(
                tokens,
                positions,
                size=size,
                sent_idx=sents,
                clip_at_sentence=clip_at_sentence,
                negation_rescue=negation_rescue,
                max_per_anchor=max_per_anchor,
            )
            kept = dedupe_overlapping(found) if dedupe else list(found)

            counter: Counter = Counter()
            for window in sorted(kept, key=lambda w: (w.anchor, w.position)):
                hit_index = counter[window.anchor]
                counter[window.anchor] += 1
                records.append(
                    window_record(
                        window, tokens, row,
                        domain=domain_of[window.anchor],
                        polysemous=polysemous_of[window.anchor],
                        size=size,
                        hit_index=hit_index,
                    )
                )

            if size == primary and label == EXHAUSTIVE_CORPUS:
                digital_counter: Counter = Counter()
                for window in sorted(found, key=lambda w: (w.anchor, w.position)):
                    if domain_of[window.anchor] != EXHAUSTIVE_DOMAIN:
                        continue
                    hit_index = digital_counter[window.anchor]
                    digital_counter[window.anchor] += 1
                    record = window_record(
                        window, tokens, row,
                        domain=EXHAUSTIVE_DOMAIN,
                        polysemous=polysemous_of[window.anchor],
                        size=size,
                        hit_index=hit_index,
                    )
                    record["url"] = row.get("url", "")
                    exhaustive.append(record)

    return records, exhaustive, docs_with_hits


def concordance_frame(frame: pd.DataFrame, source: pd.DataFrame) -> pd.DataFrame:
    """Readable KWIC view: anchor, kwic line, provenance, url when present."""
    columns = ["post_id", "anchor", "polysemous", "kwic", "sent_idx", "position",
               "negation_rescued", "truncated_left", "truncated_right", "subreddit"]
    out = frame[[c for c in columns if c in frame.columns]].copy()
    if "url" in source.columns:
        urls = source.set_index("post_id")["url"].to_dict()
        out["url"] = out["post_id"].map(urls)
    return out.sort_values(["anchor", "post_id", "position"], kind="mergesort").reset_index(drop=True)


def main() -> int:
    cfg = load_analysis_config()
    default_size = int(cfg.get("windows.size"))
    default_robustness = [int(s) for s in cfg.get("windows.robustness_sizes")]
    default_rescue = int(cfg.get("windows.negation_rescue", 4))
    default_cap = int(cfg.get("windows.max_per_anchor", 2000))
    default_dedupe = bool(cfg.get("windows.dedupe_overlap", True))

    parser = argparse.ArgumentParser(
        description="Extract +/-N token context windows around every anchor occurrence"
    )
    parser.add_argument("--corpus", choices=["a", "b", "both"], default="both")
    parser.add_argument("--analysis-dir", type=Path, default=None)
    parser.add_argument("--size", type=int, default=default_size,
                        help=f"primary window half-width in surface tokens (default {default_size})")
    parser.add_argument("--robustness-sizes", type=int, nargs="*", default=default_robustness,
                        help=f"additional half-widths to emit (default {default_robustness})")
    parser.add_argument("--no-clip-sentence", action="store_true",
                        help="let windows cross sentence boundaries (robustness check)")
    parser.add_argument("--negation-rescue", type=int, nargs="?",
                        const=default_rescue, default=default_rescue,
                        help=f"extend the left edge to a negator within N tokens (default {default_rescue})")
    parser.add_argument("--no-negation-rescue", action="store_true",
                        help="disable negation rescue entirely (equivalent to --negation-rescue 0)")
    parser.add_argument("--max-per-anchor", type=int, default=default_cap,
                        help="cap on occurrences of one anchor PER DOCUMENT (0 = no cap)")
    parser.add_argument("--no-dedupe", action="store_true",
                        help="keep overlapping windows of the same anchor")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.size < 0:
        raise SystemExit("--size must be >= 0")
    if any(s < 0 for s in args.robustness_sizes):
        raise SystemExit("--robustness-sizes must all be >= 0")

    cfg = load_analysis_config(seed=args.seed)
    anchors = load_anchors()
    analysis_dir = args.analysis_dir or cfg.path("analysis_dir")

    terms = frozenset(a.term for a in anchors.anchors)
    domain_of = {a.term: a.domain for a in anchors.anchors}
    polysemous_of = {a.term: a.polysemous for a in anchors.anchors}

    sizes = sorted({args.size, *args.robustness_sizes})
    rescue = 0 if args.no_negation_rescue else max(0, args.negation_rescue)
    clip = not args.no_clip_sentence
    cap = args.max_per_anchor if args.max_per_anchor > 0 else None
    dedupe = default_dedupe and not args.no_dedupe

    banner("Window parameters")
    item(f"token column     tokens_ctx (surface, stopwords RETAINED)")
    item(f"primary size     +/-{args.size} tokens")
    item(f"all sizes        {sizes}")
    item(f"clip at sentence {clip}")
    item(f"negation rescue  {rescue} token(s) ({len(NEGATORS)} negators)")
    item(f"dedupe overlap   {dedupe}")
    item(f"max per anchor   {cap if cap is not None else 'unlimited'} (per document)")
    item(f"anchors          {len(terms)} across {len(anchors.domains)} domains")

    labels = ["a", "b"] if args.corpus == "both" else [args.corpus]
    summary: dict[str, Any] = {
        "params": {
            "column": "tokens_ctx",
            "primary_size": args.size,
            "sizes": sizes,
            "clip_at_sentence": clip,
            "negation_rescue": rescue,
            "dedupe_overlap": dedupe,
            "max_per_anchor": cap,
            "n_anchors": len(terms),
        },
        "corpora": {},
    }

    for label in labels:
        path = analysis_dir / f"corpus_{label}.parquet"
        if not path.exists():
            raise SystemExit(f"missing {path}; run scripts/08_reprocess_tokens.py first")
        df = pd.read_parquet(path)
        missing = [c for c in ("post_id", "corpus", "subreddit", "tokens_ctx", "sent_idx")
                   if c not in df.columns]
        if missing:
            raise SystemExit(f"corpus {label}: missing column(s) {missing}")

        banner(f"Corpus {label.upper()} - {len(df)} posts")
        records, exhaustive, docs_with_hits = extract_corpus(
            df, label,
            terms=terms, domain_of=domain_of, polysemous_of=polysemous_of,
            sizes=sizes, primary=args.size,
            clip_at_sentence=clip, negation_rescue=rescue,
            max_per_anchor=cap, dedupe=dedupe,
        )
        if not records:
            item("no anchor hits; nothing written")
            summary["corpora"][label] = {"n_posts": int(len(df)), "n_windows": 0}
            continue

        windows = pd.DataFrame(records)
        destination = analysis_dir / f"windows_{label}"
        save_table(windows[list(WINDOW_COLUMNS)], destination, parquet=True, csv=False)

        primary_frame = windows[windows["window_size"] == args.size]
        item(f"posts with >=1 anchor hit  {docs_with_hits}/{len(df)} ({100 * docs_with_hits / len(df):.1f}%)")
        item(f"windows at +/-{args.size}          {len(primary_frame)}")
        item(f"windows all sizes          {len(windows)}")
        item(f"written                    data/analysis/windows_{label}.parquet")

        by_size: dict[str, Any] = {}
        for size in sizes:
            subset = windows[windows["window_size"] == size]
            n = len(subset)
            truncated = int((subset["truncated_left"] | subset["truncated_right"]).sum())
            rescued = int(subset["negation_rescued"].sum())
            by_size[str(size)] = {
                "n_windows": n,
                "mean_n_left": round(float(subset["n_left"].mean()), 3),
                "mean_n_right": round(float(subset["n_right"].mean()), 3),
                "truncation_rate": round(truncated / max(n, 1), 4),
                "truncated_left_rate": round(int(subset["truncated_left"].sum()) / max(n, 1), 4),
                "truncated_right_rate": round(int(subset["truncated_right"].sum()) / max(n, 1), 4),
                "negation_rescue_rate": round(rescued / max(n, 1), 4),
                "n_negation_rescued": rescued,
            }
            item(f"  +/-{size:<3} n={n:<6} truncated={truncated / max(n, 1):>6.1%}  "
                 f"rescued={rescued / max(n, 1):>6.1%}  "
                 f"mean span L{subset['n_left'].mean():.2f}/R{subset['n_right'].mean():.2f}")

        # ---- per-domain KWIC concordances at the primary size --------------
        banner(f"Concordances (corpus {label}, +/-{args.size})")
        written: list[str] = []
        for domain in anchors.domains:
            subset = primary_frame[primary_frame["domain"] == domain]
            if subset.empty:
                item(f"{domain:<12} 0 hits - no file written")
                continue
            table = concordance_frame(subset, df)
            paths = save_table(
                table, out_path("tables", "p3", 11, "concordance", label, domain), parquet=False
            )
            written.extend(str(p.relative_to(ROOT)) for p in paths)
            item(f"{domain:<12} {len(table):>5} lines -> {paths[0].name}")

        # ---- exhaustive Digital concordance, Corpus A ----------------------
        exhaustive_meta: dict[str, Any] | None = None
        if label == EXHAUSTIVE_CORPUS and exhaustive:
            frame = pd.DataFrame(exhaustive)
            columns = ["post_id", "url", "anchor", "polysemous", "kwic", "window_text",
                       "sent_idx", "position", "negation_rescued", "subreddit"]
            table = frame[[c for c in columns if c in frame.columns]].copy()
            table["manual_category"] = ""     # hand-coded by the author
            table = table.sort_values(["anchor", "post_id", "position"], kind="mergesort")
            path = out_path("tables", "p3", 11, "digital_concordance_exhaustive", EXHAUSTIVE_CORPUS)
            save_table(table.reset_index(drop=True), path, parquet=False)
            exhaustive_meta = {
                "n_hits": int(len(table)),
                "n_posts": int(table["post_id"].nunique()),
                "n_anchors": int(table["anchor"].nunique()),
                "by_anchor": table["anchor"].value_counts().to_dict(),
                "categories": list(MANUAL_CATEGORIES),
                "path": str(path.with_suffix(".csv").relative_to(ROOT)),
                "note": "complete enumeration, undeduped - every Digital hit in Corpus A",
            }
            banner("Exhaustive Digital concordance (Corpus A)")
            item(f"{len(table)} hits across {table['post_id'].nunique()} posts, "
                 f"{table['anchor'].nunique()} anchors - COMPLETE enumeration, not a sample")
            item(f"hand-code the empty `manual_category` column: {'/'.join(MANUAL_CATEGORIES)}")
            item(f"-> {path.with_suffix('.csv').name}")

        anchor_counts = (
            primary_frame.groupby("anchor").size().sort_values(ascending=False).to_dict()
        )
        summary["corpora"][label] = {
            "n_posts": int(len(df)),
            "n_posts_with_hits": docs_with_hits,
            "n_windows_primary": int(len(primary_frame)),
            "n_windows_all_sizes": int(len(windows)),
            "by_size": by_size,
            "by_domain": primary_frame.groupby("domain").size().to_dict(),
            "by_anchor": {str(k): int(v) for k, v in anchor_counts.items()},
            "n_anchors_with_windows": int(primary_frame["anchor"].nunique()),
            "concordances": written,
            "exhaustive_digital": exhaustive_meta,
            "path": str((analysis_dir / f"windows_{label}.parquet").relative_to(ROOT)),
        }

    save_summary(summary, out_path("summaries", "p3", 11, "windows", ext="json"))
    banner("Done")
    item("summary -> outputs/summaries/p3_11_windows.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
