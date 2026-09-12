"""Phase 5 Task 1 - tag every document with author-relative and calendar time.

T0 = the author's first recorded post in this sample (phases/New.docx). That is
a proxy for the origin event, not the true migration or loss date. Writes a
time_interval integer (quarter since T0) plus the dual calendar clock.

  python scripts/40_temporal_stratify.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analysis_config import load_analysis_config  # noqa: E402
from analysis_io import banner, item, out_path, save_summary, save_table  # noqa: E402
from temporal import cfg_temporal, tag_timelines, temporal_power_audit  # noqa: E402

KEEP_ALWAYS = (
    "post_id", "corpus", "subreddit", "username", "created_at", "text",
    "text_clean", "word_count", "tokens_lemma", "tokens_ctx", "sent_idx",
    "n_anchor_hits",
)


def _load(analysis_dir: Path, label: str) -> pd.DataFrame:
    path = analysis_dir / f"corpus_{label}.parquet"
    if not path.exists():
        raise SystemExit(f"missing {path}; run scripts/08_reprocess_tokens.py first")
    return pd.read_parquet(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Tag corpora with time_interval from each author's first post")
    parser.add_argument("--corpus", choices=["a", "b", "both"], default="both")
    args = parser.parse_args()
    cfg = load_analysis_config()
    knobs = cfg_temporal(cfg)
    analysis_dir = cfg.path("analysis_dir")
    labels = ["a", "b"] if args.corpus == "both" else [args.corpus]

    banner("Chronological stratification")
    item(f"T0            first post per username in this sample (proxy, not event date)")
    item(f"time_interval quarter since T0")
    item(f"calendar      origin {knobs['calendar_origin']}")

    audits = []
    bin_rows = []
    for label in labels:
        raw = _load(analysis_dir, label)
        tagged = tag_timelines(
            raw,
            calendar_origin=knobs["calendar_origin"],
            year1_max_months=knobs["year1_max_months"],
            year3_min_months=knobs["year3_min_months"],
        )
        tagged["corpus"] = label
        dest = analysis_dir / f"corpus_{label}_temporal.parquet"
        tagged.to_parquet(dest, index=False)
        item(f"corpus {label}   {len(tagged):,} posts -> {dest.name}")

        audit = temporal_power_audit(tagged, corpus=label)
        audits.append(audit)
        item(
            f"  authors {audit['n_authors']:,}; repeat authors {audit['n_repeat_authors']:,}; "
            f"repeat posts {audit['n_repeat_posts']:,} ({audit['repeat_post_frac']:.1%})",
            indent=2,
        )
        item(
            f"  T0 year-1 {audit['year1_posts']:,} / year-3+ {audit['year3plus_posts']:,}; "
            f"calendar year-1 {audit['cal_year1_posts']:,} / year-3+ {audit['cal_year3plus_posts']:,}",
            indent=2,
        )
        for clock, col in (("t0", "t0_quarter"), ("cal", "cal_quarter")):
            counts = tagged[col].value_counts().sort_index()
            for bin_id, n in counts.items():
                bin_rows.append({
                    "corpus": label, "clock": clock, "bin": int(bin_id), "n_posts": int(n),
                })

    save_table(pd.DataFrame(bin_rows), out_path("tables", "p5", 40, "bin_counts"))
    save_table(pd.DataFrame(audits), out_path("tables", "p5", 40, "power_audit"), parquet=False)
    save_summary(
        {
            "task": "chronological_stratification",
            "t0_definition": "author first recorded post in sample",
            "time_interval": "t0_quarter",
            "knobs": knobs,
            "audits": audits,
        },
        out_path("summaries", "p5", 40, "timelines", ext="json"),
    )
    banner("Done")
    item("Next: scripts/41_emotion_over_time.py, 42_topics_over_time.py, 43_temporal_graphs.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
