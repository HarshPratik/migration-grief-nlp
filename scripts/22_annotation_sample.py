"""Phase 3 — reproducible, balanced 150-window manual annotation sample.

Drawn from the *current* analysis corpora (emotion parquets), not the raw
window dump. Prefers non-polysemous hits so the coder is judging emotion
attached to the seed aspect, which is what Phase 3 actually asks for.

Model labels are attached as *suggestions for later agreement*, not as
answers. Leave manual_* blank until a human codes the row.

  python scripts/22_annotation_sample.py --replace
"""

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
EMOTION_CHOICES = "anger | disgust | fear | joy | neutral | sadness | surprise"
PROB_COLS = [
    "p_anger",
    "p_disgust",
    "p_fear",
    "p_joy",
    "p_neutral",
    "p_sadness",
    "p_surprise",
]


def _emotion_paths(cfg) -> list[Path]:
    analysis = cfg.path("analysis_dir")
    model = str(cfg.get("emotion.model")).split("/")[-1]
    paths = [analysis / f"emotion_{c}_{model}.parquet" for c in ("a", "b")]
    if all(p.exists() for p in paths):
        return paths
    # Fall back to any DistilRoBERTa-named file in the analysis dir.
    found = sorted(analysis.glob("emotion_*_emotion-english-distilroberta-base.parquet"))
    if len(found) >= 2:
        return found
    return paths


def _load_windows(cfg) -> pd.DataFrame:
    size = int(cfg.get("windows.size"))
    paths = _emotion_paths(cfg)
    frames = []
    for path in paths:
        if not path.exists():
            raise SystemExit(
                f"missing {path}; run scripts/15_emotion_absa.py on the 10k corpora first"
            )
        frame = pd.read_parquet(path)
        if "window_size" in frame.columns:
            frame = frame[frame["window_size"] == size]
        frames.append(frame)
    windows = pd.concat(frames, ignore_index=True)
    if windows.empty:
        raise SystemExit("no size-matched emotion windows to sample from")
    return windows


def _stratum_sample(group: pd.DataFrame, count: int, rng) -> pd.DataFrame:
    """Prefer non-polysemous windows; fall back if a stratum is too thin."""
    if "polysemous" in group.columns:
        strict = group[group["polysemous"] == False]  # noqa: E712
        if len(strict) >= count:
            group = strict
    if len(group) < count:
        raise SystemExit(
            f"stratum has only {len(group)} rows; need {count}"
        )
    chosen = rng.choice(group.index.to_numpy(), size=count, replace=False)
    return group.loc[chosen]


def main() -> int:
    parser = argparse.ArgumentParser(description="Create balanced manual annotation sample")
    parser.add_argument(
        "--replace",
        action="store_true",
        help="replace an existing blank annotation template",
    )
    args = parser.parse_args()

    cfg = load_analysis_config()
    output = out_path("tables", "p3", 22, "emotion_annotation_sample150")
    if output.exists() and not args.replace:
        raise SystemExit(
            f"annotation file already exists: {output}; refusing to overwrite possible manual coding"
        )

    windows = _load_windows(cfg)
    strata = sorted(windows.groupby(["corpus", "domain"]).groups)
    if len(strata) != 8:
        raise SystemExit(f"expected 8 corpus×domain strata, got {strata}")
    base, remainder = divmod(SAMPLE_SIZE, len(strata))
    rng = make_rng(cfg.seed + 22)
    sampled = []
    allocation = {}
    for index, stratum in enumerate(strata):
        group = windows[(windows["corpus"] == stratum[0]) & (windows["domain"] == stratum[1])]
        count = base + (1 if index < remainder else 0)
        chunk = _stratum_sample(group, count, rng)
        allocation[f"{stratum[0]}/{stratum[1]}"] = {
            "requested": count,
            "n_available": int(len(group)),
            "n_non_polysemous": int((group["polysemous"] == False).sum())  # noqa: E712
            if "polysemous" in group.columns
            else None,
            "sampled_non_polysemous": int((chunk["polysemous"] == False).sum())  # noqa: E712
            if "polysemous" in chunk.columns
            else None,
        }
        sampled.append(chunk)

    table = pd.concat(sampled, ignore_index=True).sample(frac=1, random_state=cfg.seed).reset_index(drop=True)
    table.insert(0, "annotation_id", [f"E{i:03d}" for i in range(1, len(table) + 1)])
    table["manual_category"] = ""
    table["manual_emotion"] = ""
    table["confidence_1_to_3"] = ""
    table["coder_notes"] = ""
    table["category_guide"] = CATEGORIES
    table["emotion_guide"] = EMOTION_CHOICES
    if "emotion" in table.columns:
        table["model_emotion"] = table["emotion"]
    else:
        table["model_emotion"] = ""
    if all(c in table.columns for c in PROB_COLS):
        table["model_p_max"] = table[PROB_COLS].max(axis=1).round(4)
    else:
        table["model_p_max"] = ""

    columns = [
        "annotation_id",
        "corpus",
        "domain",
        "anchor",
        "window_text",
        "polysemous",
        "negation_rescued",
        "manual_category",
        "manual_emotion",
        "confidence_1_to_3",
        "coder_notes",
        "category_guide",
        "emotion_guide",
        "model_emotion",
        "model_p_max",
        "post_id",
    ]
    for col in columns:
        if col not in table.columns:
            table[col] = ""
    save_table(table[columns], output, parquet=False)
    save_summary(
        {
            "n_windows": len(table),
            "allocation": "balanced across corpus x domain strata",
            "sampling": "non-polysemous preferred; drawn from 10k emotion windows",
            "categories": CATEGORIES.split(" | "),
            "emotion_labels": EMOTION_CHOICES.split(" | "),
            "model_labels": "suggestions only — do not copy into manual_emotion",
            "stratum_allocation": allocation,
            "privacy": "Restricted: contains short excerpts and post identifiers; excluded from shareable bundle.",
        },
        out_path("summaries", "p3", 22, "emotion_annotation_sample", ext="json"),
    )
    banner("Annotation sample")
    item(f"wrote {output.relative_to(ROOT)} ({len(table)} rows; restricted)")
    item("Fill manual_category + manual_emotion; ignore model_* until agreement")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
