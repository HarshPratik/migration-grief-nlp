"""Phase 3 - domain emotion distribution bars and corpus-difference heatmap."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analysis_config import load_analysis_config, load_anchors  # noqa: E402
from analysis_io import banner, item, out_path, save_summary  # noqa: E402
from plotting import apply_style, fig_heatmap, fig_stacked_bar, save_fig  # noqa: E402


def main() -> int:
    cfg = load_analysis_config()
    model_name = str(cfg.get("emotion.model")).split("/")[-1]
    scored = []
    for corpus in ("a", "b"):
        path = cfg.path("analysis_dir") / f"emotion_{corpus}_{model_name}.parquet"
        if not path.exists():
            raise SystemExit(f"missing {path}; run scripts/15_emotion_absa.py first")
        scored.append(pd.read_parquet(path))
    windows = pd.concat(scored, ignore_index=True)
    anchors = load_anchors()
    polysemous = {anchor.term for anchor in anchors.anchors if anchor.polysemous}
    metadata = []
    for corpus in ("a", "b"):
        frame = pd.read_parquet(
            cfg.path("analysis_dir") / f"corpus_{corpus}.parquet",
            columns=["post_id", "username"],
        )
        metadata.append(frame.assign(corpus=corpus))
    retained = (
        pd.concat(metadata, ignore_index=True)
        .sort_values("post_id")
        .drop_duplicates(["corpus", "username"])
    )
    windows = windows[
        windows["post_id"].astype(str).isin(set(retained["post_id"].astype(str)))
        & ~windows["anchor"].isin(polysemous)
    ]
    probability_columns = [column for column in windows.columns if column.startswith("p_")]
    emotions = [column.removeprefix("p_") for column in probability_columns]
    domains = (
        windows.groupby(["corpus", "domain", "post_id"])[probability_columns]
        .mean()
        .groupby(["corpus", "domain"])
        .mean()
        .reset_index()
        .pivot(index="domain", columns="corpus", values=probability_columns)
    )
    if domains.empty:
        raise SystemExit("emotion scores have no domain rows")
    style = apply_style(cfg.section("figures"))
    labels = cfg.get("figures.corpus_labels")
    colors = {str(key): str(value) for key, value in cfg.get("figures.emotion_colors").items()}
    categories = []
    layers = {emotion: [] for emotion in emotions}
    for domain, row in domains.iterrows():
        for corpus in ("a", "b"):
            categories.append(f"{domain}\n{labels[corpus]}")
            for emotion in emotions:
                layers[emotion].append(float(row[(f"p_{emotion}", corpus)]))

    fig, _ = fig_stacked_bar(
        categories, layers, style=style,
        title="Emotion probabilities in anchor-context windows, post-aggregated",
        ylabel="mean predicted probability", colors=colors,
    )
    written = save_fig(fig, out_path("figures", "p3", 19, "emotion_stacked_domain"),
                       formats=list(cfg.get("figures.formats")))

    differences = np.array([
        [float(row[(f"p_{emotion}", "a")]) - float(row[(f"p_{emotion}", "b")]) for emotion in emotions]
        for _, row in domains.iterrows()
    ])
    bound = max(float(np.abs(differences).max()), 0.01)
    fig, _ = fig_heatmap(
        differences, domains.index.astype(str).tolist(), emotions,
        style=style,
        title="Emotion difference: Corpus A minus Corpus B",
        cbar_label="difference in mean probability",
        cmap=str(cfg.get("figures.diverging_cmap")), vmin=-bound, vmax=bound,
    )
    written.extend(save_fig(fig, out_path("figures", "p3", 19, "emotion_difference_heatmap"),
                            formats=list(cfg.get("figures.formats"))))

    save_summary({
        "n_domains": len(domains),
        "emotions": emotions,
        "note": "Probabilities are averaged within post before corpus comparison.",
        "figures": [str(path) for path in written],
    }, out_path("summaries", "p3", 19, "emotion_figures", ext="json"))
    banner("Emotion figures")
    for output in written:
        item(f"wrote {output.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
