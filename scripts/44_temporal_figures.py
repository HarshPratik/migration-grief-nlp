"""Phase 5 figures: sadness over time, topics over time, betweenness spikes.

  python scripts/44_temporal_figures.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analysis_config import load_analysis_config  # noqa: E402
from analysis_io import banner, item, out_path  # noqa: E402
from plotting import apply_style, new_figure, save_fig  # noqa: E402


def _copy_docs(path: Path) -> None:
    dest = ROOT / "docs" / "figures" / path.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, dest)


def _sadness_lines(series: pd.DataFrame, style: dict, clock: str, source: str) -> Path | None:
    sub = series[
        (series["emotion"] == "sadness")
        & (series["domain"] == "all")
        & (series["sample"] == "all")
        & (series["clock"] == clock)
        & (series["source"] == source)
    ]
    if sub.empty:
        return None
    fig, ax = new_figure(style)
    colors = {"a": style.get("corpus_a", "#4C78A8"), "b": style.get("corpus_b", "#E45756")}
    labels = {"a": "A migration (stationarity hyp.)", "b": "B grief (decay hyp.)"}
    for corpus, grp in sub.groupby("corpus"):
        g = grp.sort_values("bin")
        ax.plot(g["bin"], g["mean"], marker="o", label=labels.get(corpus, corpus), color=colors.get(corpus))
        se = (g["std"] / np.sqrt(g["n"].clip(lower=1))).to_numpy()
        ax.fill_between(g["bin"], g["mean"] - se, g["mean"] + se, color=colors.get(corpus), alpha=0.15)
    ax.set_xlabel(f"{clock} quarter bin")
    ax.set_ylabel("mean P(sadness)")
    ax.set_title(f"Sadness over time ({source}, {clock} clock)")
    ax.legend()
    path = out_path("figures", "p5", 44, "sadness_over_time", clock, source, ext="png")
    save_fig(fig, path)
    _copy_docs(path)
    return path


def _topic_lines(prev: pd.DataFrame, style: dict, corpus: str, clock: str) -> Path | None:
    sub = prev[(prev["corpus"] == corpus) & (prev["clock"] == clock)]
    if sub.empty:
        return None
    fig, ax = new_figure(style, width=9.0, height=5.5)
    for topic, grp in sub.groupby("topic"):
        g = grp.sort_values("bin")
        ax.plot(g["bin"], g["share"], label=f"t{int(topic)}", linewidth=1.4)
    ax.set_xlabel(f"{clock} quarter")
    ax.set_ylabel("topic share")
    ax.set_title(f"Topics over time — corpus {corpus.upper()} ({clock})")
    ax.legend(ncol=4, fontsize=8)
    path = out_path("figures", "p5", 44, "topics_over_time", corpus, clock, ext="png")
    save_fig(fig, path)
    _copy_docs(path)
    return path


def _spike_bars(delta: pd.DataFrame, style: dict, corpus: str, clock: str) -> Path | None:
    sub = delta[(delta["corpus"] == corpus) & (delta["clock"] == clock)].copy()
    if sub.empty:
        return None
    sub = sub.sort_values("delta", ascending=False).head(20)
    fig, ax = new_figure(style, width=8.5, height=6.0)
    colors = ["#54A24B" if bool(h) else "#4C78A8" for h in sub["hybrid"]]
    ax.barh(sub["node"][::-1], sub["delta"][::-1], color=list(reversed(colors)))
    ax.set_xlabel("betweenness late − early")
    ax.set_title(f"Betweenness change — corpus {corpus.upper()} ({clock})")
    path = out_path("figures", "p5", 44, "betweenness_delta", corpus, clock, ext="png")
    save_fig(fig, path)
    _copy_docs(path)
    return path


def main() -> int:
    cfg = load_analysis_config()
    style = apply_style(cfg.section("figures"))
    style["corpus_a"] = cfg.get("figures.corpus_colors.a")
    style["corpus_b"] = cfg.get("figures.corpus_colors.b")
    banner("Phase 5 figures")
    written = []

    emotion_path = out_path("tables", "p5", 41, "emotion_bin_means", ext="csv")
    if emotion_path.exists():
        series = pd.read_csv(emotion_path)
        for clock in ("t0", "calendar"):
            for source in ("window", "fullpost"):
                p = _sadness_lines(series, style, clock, source)
                if p:
                    written.append(p)
                    item(str(p.name))

    topic_path = out_path("tables", "p5", 42, "topics_over_time", ext="csv")
    if topic_path.exists():
        prev = pd.read_csv(topic_path)
        for corpus in ("a", "b"):
            for clock in ("t0", "calendar"):
                p = _topic_lines(prev, style, corpus, clock)
                if p:
                    written.append(p)
                    item(str(p.name))

    delta_path = out_path("tables", "p5", 43, "betweenness_delta", ext="csv")
    if delta_path.exists():
        delta = pd.read_csv(delta_path)
        for corpus in ("a", "b"):
            for clock in ("t0", "calendar"):
                p = _spike_bars(delta, style, corpus, clock)
                if p:
                    written.append(p)
                    item(str(p.name))

    item(f"wrote {len(written)} figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
