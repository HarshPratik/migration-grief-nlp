"""Phases 3-4 - exploratory network and topic stability/prevalence figures."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analysis_config import load_analysis_config  # noqa: E402
from analysis_io import banner, item, out_path, save_summary  # noqa: E402
from plotting import apply_style, fig_grouped_bar, fig_network, save_fig  # noqa: E402


DOMAIN_COLORS = {
    "temporal": "#4C78A8",
    "identity": "#72B7B2",
    "absence": "#E45756",
    "digital": "#F58518",
}


def _seed_domain_graph(corpus: str) -> nx.Graph:
    """Union of per-domain PPMI subgraphs so the figure is about the seed dictionary."""
    merged = nx.Graph()
    for domain in ("temporal", "identity", "absence", "digital"):
        path = out_path("networks", "p3", 16, "cooccur", corpus, domain, ext="graphml")
        if not path.exists():
            continue
        part = nx.read_graphml(path)
        for node, data in part.nodes(data=True):
            attrs = dict(data)
            attrs["domain"] = domain
            merged.add_node(node, **attrs)
        merged.add_edges_from(part.edges(data=True))
    return merged


def network_figures(cfg, style) -> list[Path]:
    written: list[Path] = []
    for corpus in ("a", "b"):
        graph = _seed_domain_graph(corpus)
        if graph.number_of_nodes() == 0:
            path = out_path("networks", "p3", 16, "cooccur", corpus, ext="graphml")
            if not path.exists():
                raise SystemExit(f"missing {path}; run scripts/16_cooccurrence.py first")
            graph = nx.read_graphml(path)
        sizes = {
            node: 80.0 + 28.0 * np.log1p(float(data.get("freq", data.get("count", 1))))
            for node, data in graph.nodes(data=True)
        }
        node_colors = {
            node: DOMAIN_COLORS.get(str(data.get("domain", "")), "#4C78A8")
            for node, data in graph.nodes(data=True)
        }
        fig, _ = fig_network(
            graph, style=style,
            title=f"Corpus {corpus.upper()} seed-domain co-occurrence (PPMI)",
            node_sizes=sizes,
            node_colors=node_colors,
            layout="spring",
            layout_seed=int(cfg.get("cooccurrence.layout_seed")),
            layout_iterations=int(cfg.get("cooccurrence.layout_iterations")),
            label_top_n=40,
            color_legend={
                "Temporal": DOMAIN_COLORS["temporal"],
                "Identity": DOMAIN_COLORS["identity"],
                "Absence": DOMAIN_COLORS["absence"],
                "Digital": DOMAIN_COLORS["digital"],
            },
            max_nodes=80,
            max_edges=120,
        )
        written.extend(save_fig(fig, out_path("figures", "p3", 20, "cooccurrence", corpus),
                                formats=list(cfg.get("figures.formats"))))
        dest = ROOT / "docs" / "figures" / f"p3_20_cooccurrence__{corpus}.png"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(out_path("figures", "p3", 20, "cooccurrence", corpus).with_suffix(".png").read_bytes())
    return written


def topic_figures(cfg, style) -> list[Path]:
    method = str(cfg.get("topics.method"))
    summary_path = out_path("summaries", "p4", 17, "topic_model", ext="json")
    if not summary_path.exists():
        raise SystemExit(f"missing {summary_path}; run scripts/17_topic_model.py first")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if str(summary.get("method")) != method:
        raise SystemExit(
            f"topic summary method {summary.get('method')!r} does not match configured {method!r}"
        )
    k = int(summary["k"])
    written: list[Path] = []

    sweep_path = out_path("tables", "p4", 17, "k_sweep", method)
    if sweep_path.exists():
        sweep = pd.read_csv(sweep_path)
        if not sweep.empty:
            fig, ax = fig_grouped_bar(
                [str(value) for value in sweep["k"]],
                {
                    "matched agreement": sweep["mean_agreement"].tolist(),
                    "non-matched baseline": sweep["null_agreement"].tolist(),
                    "core-topic fraction": sweep["core_topic_fraction"].tolist(),
                },
                style=style, title="Topic stability across candidate k", ylabel="score",
            )
            ax.set_xlabel("number of topics (k)")
            written.extend(save_fig(fig, out_path("figures", "p4", 20, "topic_stability"),
                                    formats=list(cfg.get("figures.formats"))))
    prevalence_path = out_path("tables", "p4", 17, "topic_prevalence", f"{method}_k{k}")
    prevalence = pd.read_csv(prevalence_path)
    labels = [f"T{int(row.topic)}{' *' if bool(row.is_core) else ''}" for row in prevalence.itertuples()]
    corpus_labels = cfg.get("figures.corpus_labels")
    colors = cfg.get("figures.corpus_colors")
    fig, ax = fig_grouped_bar(
        labels,
        {
            str(corpus_labels["a"]): prevalence["mean_theta_a"].tolist(),
            str(corpus_labels["b"]): prevalence["mean_theta_b"].tolist(),
        },
        style=style,
        title="Topic reconstruction-loading share (* = reproducible core topic)",
        ylabel="mean normalized NMF loading",
        colors={str(corpus_labels[key]): str(colors[key]) for key in ("a", "b")},
    )
    ax.text(0.99, 0.01, "Only starred topics meet the recurrence threshold.",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8)
    written.extend(save_fig(fig, out_path("figures", "p4", 20, "topic_prevalence"),
                            formats=list(cfg.get("figures.formats"))))
    return written


def main() -> int:
    cfg = load_analysis_config()
    style = apply_style(cfg.section("figures"))
    written = network_figures(cfg, style) + topic_figures(cfg, style)
    save_summary({
        "figures": [str(path) for path in written],
        "network_note": "Top-edge PPMI networks are exploratory; many exported edges have low raw counts.",
        "topic_note": "Only topics meeting the configured recurrence threshold are reportable as core.",
    }, out_path("summaries", "p4", 20, "network_topic_figures", ext="json"))
    banner("Network and topic figures")
    for output in written:
        item(f"wrote {output.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
