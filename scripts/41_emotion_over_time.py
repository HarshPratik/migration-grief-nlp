"""Phase 5 Task 2 - sadness over time (asymmetrical mourning + extras).

New.docx: mean DistilRoBERTa sadness per temporal bin; linregress slope.
Decay (negative, FDR-significant) vs stationarity (null / near-zero).

Paper-strength extras kept on the same model and corpora:
- dual clocks (author-relative T0 vs calendar)
- repeat-author subsample vs all posts
- post-level OLS with author-clustered bootstrap, not only bin means
- Theil-Sen robust slope
- permutation p-values
- all seven emotion classes
- domain-stratified window emotion (Phase 3 continuity)
- optional full-post classification (document-level, not ABSA windows)

  python scripts/41_emotion_over_time.py
  python scripts/41_emotion_over_time.py --skip-fullpost
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analysis_config import load_analysis_config, load_anchors  # noqa: E402
from analysis_io import banner, item, make_rng, out_path, save_summary, save_table  # noqa: E402
from temporal import (  # noqa: E402
    benjamini_hochberg,
    bin_means,
    cfg_temporal,
    cluster_bootstrap_slope,
    interpret_sadness_slope,
    linregress_row,
    permute_slope_pvalue,
    theil_sen_row,
)

EMOTIONS = ("anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise")
MODEL_FILE = "emotion-english-distilroberta-base"


def classify_full_posts(texts: list[str], model_name: str, batch_size: int, max_length: int) -> tuple[np.ndarray, list[str]]:
    """Full-post probabilities; uses CUDA when available."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    labels = [model.config.id2label[i].lower() for i in range(model.config.num_labels)]
    out = np.zeros((len(texts), len(labels)), dtype=np.float32)
    for start in range(0, len(texts), batch_size):
        chunk = texts[start: start + batch_size]
        encoded = tokenizer(
            chunk, return_tensors="pt", padding=True, truncation=True, max_length=max_length,
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            logits = model(**encoded).logits
        out[start: start + len(chunk)] = torch.softmax(logits, dim=-1).cpu().numpy()
    return out, labels


def _load_temporal(analysis_dir: Path, label: str) -> pd.DataFrame:
    path = analysis_dir / f"corpus_{label}_temporal.parquet"
    if not path.exists():
        raise SystemExit(f"missing {path}; run scripts/40_temporal_stratify.py first")
    cols = [
        "post_id", "corpus", "username", "created_at", "text", "word_count",
        "months_since_t0", "time_interval", "t0_quarter", "cal_quarter",
        "cal_months", "is_repeat_author", "n_posts_author",
    ]
    return pd.read_parquet(path, columns=cols)


def _load_window_emotion(analysis_dir: Path, label: str, window_size: int) -> pd.DataFrame:
    path = analysis_dir / f"emotion_{label}_{MODEL_FILE}.parquet"
    if not path.exists():
        raise SystemExit(f"missing {path}; run scripts/15_emotion_absa.py first")
    frame = pd.read_parquet(path)
    if "window_size" in frame.columns:
        frame = frame[frame["window_size"] == window_size]
    return frame


def _post_emotion(windows: pd.DataFrame, emotion_cols: list[str], *, domain: str | None) -> pd.DataFrame:
    subset = windows if domain is None else windows[windows["domain"] == domain]
    if subset.empty:
        return pd.DataFrame()
    grouped = subset.groupby("post_id", as_index=False)[emotion_cols].mean()
    grouped["n_windows"] = subset.groupby("post_id").size().to_numpy()
    if domain is not None:
        grouped["domain"] = domain
    else:
        grouped["domain"] = "all"
    return grouped


def _slope_bundle(
    frame: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    knobs: dict,
    rng: np.random.Generator,
    bootstrap: bool,
) -> dict[str, float]:
    ols = linregress_row(frame[x_col].to_numpy(), frame[y_col].to_numpy())
    robust = theil_sen_row(frame[x_col].to_numpy(), frame[y_col].to_numpy())
    if bootstrap:
        boot = cluster_bootstrap_slope(
            frame, x_col, y_col, "username",
            n_boot=knobs["n_bootstrap"], rng=rng,
        )
        boot_slope, lo, hi = boot["slope"], boot["ci_low"], boot["ci_high"]
    else:
        boot_slope, lo, hi = ols["slope"], np.nan, np.nan
    return {
        **{f"ols_{k}": v for k, v in ols.items()},
        "theil_slope": robust["slope"],
        "theil_low": robust["low"],
        "theil_high": robust["high"],
        "perm_p": np.nan,
        "boot_slope": boot_slope,
        "boot_ci_low": lo,
        "boot_ci_high": hi,
    }


def _bin_slope(means: pd.DataFrame, knobs: dict, rng: np.random.Generator) -> dict[str, float]:
    if len(means) < 3:
        return {
            "ols_n": float(len(means)), "ols_slope": np.nan, "ols_pvalue": np.nan,
            "ols_stderr": np.nan, "ols_rvalue": np.nan, "ols_intercept": np.nan,
            "theil_slope": np.nan, "theil_low": np.nan, "theil_high": np.nan,
            "perm_p": np.nan, "boot_slope": np.nan, "boot_ci_low": np.nan, "boot_ci_high": np.nan,
        }
    ols = linregress_row(means["bin"].to_numpy(), means["mean"].to_numpy())
    robust = theil_sen_row(means["bin"].to_numpy(), means["mean"].to_numpy())
    perm = permute_slope_pvalue(
        means["bin"].to_numpy(), means["mean"].to_numpy(),
        n_perm=knobs["n_permutations"], rng=rng,
    )
    return {
        **{f"ols_{k}": v for k, v in ols.items()},
        "theil_slope": robust["slope"],
        "theil_low": robust["low"],
        "theil_high": robust["high"],
        "perm_p": perm,
        "boot_slope": ols["slope"],
        "boot_ci_low": np.nan,
        "boot_ci_high": np.nan,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Emotion decay vs stationarity over temporal bins")
    parser.add_argument("--skip-fullpost", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="debug: first N posts per corpus for full-post")
    args = parser.parse_args()

    cfg = load_analysis_config()
    anchors = load_anchors()
    knobs = cfg_temporal(cfg)
    rng = make_rng(cfg.seed)
    analysis_dir = cfg.path("analysis_dir")
    window_size = int(cfg.get("windows.size"))
    model_name = str(cfg.get("emotion.model"))
    batch_size = int(cfg.get("emotion.batch_size"))
    domains = ["all", *list(anchors.domains.keys())]
    polysemous = {a.term for a in anchors.anchors if a.polysemous}

    banner("Emotion over time")
    item(f"model     {model_name}")
    item("T0        first post in sample (proxy)")
    item("roles     A=migrant (stationarity hyp.); B=grief comparandum (decay hyp.)")

    series_rows = []
    slope_rows = []
    for label in ("a", "b"):
        posts = _load_temporal(analysis_dir, label)
        windows = _load_window_emotion(analysis_dir, label, window_size)
        windows = windows[~windows["anchor"].isin(polysemous)].copy()
        emotion_cols = [c for c in (f"p_{e}" for e in EMOTIONS) if c in windows.columns]
        if "p_sadness" not in emotion_cols:
            raise SystemExit(f"corpus {label}: window emotion is missing p_sadness")

        item(f"corpus {label}  posts {len(posts):,}  windows {len(windows):,}")

        for domain in domains:
            per_post = _post_emotion(windows, emotion_cols, domain=None if domain == "all" else domain)
            if per_post.empty:
                continue
            merged = posts.merge(per_post, on="post_id", how="inner")
            for sample_name, mask in (
                ("all", np.ones(len(merged), dtype=bool)),
                ("repeat", merged["is_repeat_author"].to_numpy()),
            ):
                sub = merged.loc[mask].copy()
                if sub.empty:
                    continue
                for clock, bin_col, month_col in (
                    ("t0", "t0_quarter", "months_since_t0"),
                    ("calendar", "cal_quarter", "cal_months"),
                ):
                    for emotion in EMOTIONS:
                        col = f"p_{emotion}"
                        if col not in sub.columns:
                            continue
                        means = bin_means(
                            sub, bin_col=bin_col, value_col=col,
                            min_bin_posts=knobs["min_bin_posts"],
                        )
                        for _, row in means.iterrows():
                            series_rows.append({
                                "corpus": label, "clock": clock, "sample": sample_name,
                                "domain": domain, "emotion": emotion, "source": "window",
                                "bin": int(row["bin"]), "mean": float(row["mean"]),
                                "std": float(row["std"]) if pd.notna(row["std"]) else np.nan,
                                "n": int(row["n"]),
                            })
                        if emotion != "sadness" and domain != "all":
                            continue
                        if emotion != "sadness" and sample_name != "all":
                            continue
                        bin_fit = _bin_slope(means, knobs, rng)
                        post_fit = _slope_bundle(
                            sub, x_col=month_col, y_col=col, knobs=knobs, rng=rng,
                            bootstrap=(emotion == "sadness" and domain == "all"),
                        )
                        for unit, fit in (("bin_mean", bin_fit), ("post_ols", post_fit)):
                            slope_rows.append({
                                "corpus": label, "clock": clock, "sample": sample_name,
                                "domain": domain, "emotion": emotion, "source": "window",
                                "unit": unit, "n_posts": int(len(sub)), "n_bins": int(len(means)),
                                **fit,
                            })

        if args.skip_fullpost:
            continue
        texts = posts["text"].fillna("").astype(str).tolist()
        if args.limit:
            posts = posts.head(args.limit).copy()
            texts = texts[: args.limit]
        item(f"  full-post inference on {len(texts):,} documents ({knobs['fullpost_max_length']} tokens)")
        probs, labels = classify_full_posts(
            texts, model_name, batch_size, knobs["fullpost_max_length"],
        )
        scored = posts.copy()
        for i, name in enumerate(labels):
            scored[f"p_{name}"] = probs[:, i]
        save_table(
            scored[["post_id", "corpus", "username", *[f"p_{n}" for n in labels if f"p_{n}" in scored]]],
            analysis_dir / f"emotion_fullpost_{label}_{MODEL_FILE}",
            parquet=True, csv=False,
        )
        for sample_name, mask in (
            ("all", np.ones(len(scored), dtype=bool)),
            ("repeat", scored["is_repeat_author"].to_numpy()),
        ):
            sub = scored.loc[mask].copy()
            if sub.empty:
                continue
            for clock, bin_col, month_col in (
                ("t0", "t0_quarter", "months_since_t0"),
                ("calendar", "cal_quarter", "cal_months"),
            ):
                col = "p_sadness"
                means = bin_means(sub, bin_col=bin_col, value_col=col, min_bin_posts=knobs["min_bin_posts"])
                for _, row in means.iterrows():
                    series_rows.append({
                        "corpus": label, "clock": clock, "sample": sample_name,
                        "domain": "all", "emotion": "sadness", "source": "fullpost",
                        "bin": int(row["bin"]), "mean": float(row["mean"]),
                        "std": float(row["std"]) if pd.notna(row["std"]) else np.nan,
                        "n": int(row["n"]),
                    })
                bin_fit = _bin_slope(means, knobs, rng)
                post_fit = _slope_bundle(
                    sub, x_col=month_col, y_col=col, knobs=knobs, rng=rng,
                    bootstrap=True,
                )
                for unit, fit in (("bin_mean", bin_fit), ("post_ols", post_fit)):
                    slope_rows.append({
                        "corpus": label, "clock": clock, "sample": sample_name,
                        "domain": "all", "emotion": "sadness", "source": "fullpost",
                        "unit": unit, "n_posts": int(len(sub)), "n_bins": int(len(means)),
                        **fit,
                    })

    series = pd.DataFrame(series_rows)
    slopes = pd.DataFrame(slope_rows)
    primary = (
        (slopes["emotion"] == "sadness")
        & (slopes["domain"] == "all")
        & (slopes["source"] == "window")
        & (slopes["unit"] == "bin_mean")
        & (slopes["sample"] == "all")
    )
    slopes["q_ols"] = np.nan
    if primary.any():
        q = benjamini_hochberg(slopes.loc[primary, "ols_pvalue"].to_numpy())
        slopes.loc[primary, "q_ols"] = q
    slopes["verdict"] = [
        interpret_sadness_slope(s, p, q if pd.notna(q) else 1.0)
        for s, p, q in zip(slopes["ols_slope"], slopes["ols_pvalue"], slopes["q_ols"])
    ]
    # New.docx mapping: A should be stationary, B should decay — recorded, not forced.
    role = {"a": "migrant_stationarity_hyp", "b": "grief_decay_hyp"}
    slopes["hypothesis"] = slopes["corpus"].map(role)

    save_table(series, out_path("tables", "p5", 41, "emotion_bin_means"))
    save_table(slopes, out_path("tables", "p5", 41, "emotion_slopes"))
    primary_tbl = slopes[primary].copy()
    save_table(primary_tbl, out_path("tables", "p5", 41, "sadness_primary"), parquet=False)
    save_summary(
        {
            "task": "emotion_decay_stationarity",
            "model": model_name,
            "t0": "first post in sample",
            "primary_family": "window sadness, all posts, bin-mean, both clocks, both corpora",
            "n_slope_rows": int(len(slopes)),
            "primary": primary_tbl.to_dict(orient="records"),
        },
        out_path("summaries", "p5", 41, "emotion_time", ext="json"),
    )
    banner("Done")
    item(f"slope tests {len(slopes)}  primary family {int(primary.sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
