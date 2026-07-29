"""Phase 3 - reproducible, balanced 150-window manual annotation sample."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analysis_config import load_analysis_config  # noqa: E402
from analysis_io import banner, item, make_rng, out_path, save_summary, save_table  # noqa: E402

SAMPLE_SIZE = 150
CATEGORIES = "aspect_owned | surrounding_context | ambiguous | wrong_anchor_match"


def main() -> int:
    parser = argparse.ArgumentParser(description="Create balanced manual annotation sample")
    parser.add_argument("--replace", action="store_true",
                        help="replace an existing blank annotation template")
    args = parser.parse_args()

    cfg = load_analysis_config()
    output = out_path("tables", "p3", 22, "emotion_annotation_sample150")
    if output.exists() and not args.replace:
        raise SystemExit(
            f"annotation file already exists: {output}; refusing to overwrite possible manual coding"
        )
    frames = []
    for corpus in ("a", "b"):
        path = cfg.path("analysis_dir") / f"windows_{corpus}.parquet"
        if not path.exists():
            raise SystemExit(f"missing {path}; run scripts/11_extract_windows.py first")
        frame = pd.read_parquet(path)
        frames.append(frame[frame["window_size"] == int(cfg.get("windows.size"))])
    windows = pd.concat(frames, ignore_index=True)

    # Equal allocation across the eight corpus-domain strata. The remainder is
    # assigned in sorted order, making the exact 150-row sample reproducible.
    strata = sorted(windows.groupby(["corpus", "domain"]).groups)
    base, remainder = divmod(SAMPLE_SIZE, len(strata))
    rng = make_rng(cfg.seed + 22)
    sampled = []
    for index, stratum in enumerate(strata):
        group = windows[(windows["corpus"] == stratum[0]) & (windows["domain"] == stratum[1])]
        count = base + (1 if index < remainder else 0)
        if len(group) < count:
            raise SystemExit(f"stratum {stratum} has only {len(group)} rows; need {count}")
        chosen = rng.choice(group.index.to_numpy(), size=count, replace=False)
        sampled.append(group.loc[chosen])

    table = pd.concat(sampled, ignore_index=True).sample(frac=1, random_state=cfg.seed).reset_index(drop=True)
    table.insert(0, "annotation_id", [f"E{i:03d}" for i in range(1, len(table) + 1)])
    table["manual_category"] = ""
    table["manual_emotion"] = ""
    table["confidence_1_to_3"] = ""
    table["coder_notes"] = ""
    table["category_guide"] = CATEGORIES

    columns = [
        "annotation_id", "corpus", "domain", "anchor", "window_text",
        "negation_rescued", "manual_category", "manual_emotion",
        "confidence_1_to_3", "coder_notes", "category_guide", "post_id",
    ]
    save_table(table[columns], output, parquet=False)
    save_summary({
        "n_windows": len(table),
        "allocation": "balanced across corpus x domain strata",
        "categories": CATEGORIES.split(" | "),
        "privacy": "Restricted: contains short excerpts and post identifiers; excluded from shareable bundle.",
    }, out_path("summaries", "p3", 22, "emotion_annotation_sample", ext="json"))
    banner("Annotation sample")
    item(f"wrote {output.relative_to(ROOT)} ({len(table)} rows; restricted)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
