"""Phases 2-4 - consistent figure styling and reusable chart builders.

Shared module for scripts 18-20. Every figure in the paper goes through here so
the ~40 outputs read as one document: one palette, one font, one DPI.

Uses the Agg backend, set at import, so scripts render identically headless and
never try to open a window on a laptop or in a scheduled run.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")  # must precede pyplot import

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

DEFAULT_STYLE: dict[str, Any] = {
    "dpi": 300,
    "width_in": 8.0,
    "height_in": 5.0,
    "font_family": "DejaVu Sans",  # ships with matplotlib; no Windows font risk
    "base_font_size": 11,
    "heatmap_cmap": "viridis",
    "diverging_cmap": "RdBu_r",
    "annotate_max_cells": 400,
}


def apply_style(figures_config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Install global rcParams from config. Returns the resolved style dict."""
    style = dict(DEFAULT_STYLE)
    if figures_config:
        style.update({k: v for k, v in figures_config.items() if v is not None})

    plt.rcParams.update({
        "figure.dpi": 100,                       # on-screen; savefig uses style["dpi"]
        "savefig.dpi": style["dpi"],
        "savefig.bbox": "tight",
        "font.family": style["font_family"],
        "font.size": style["base_font_size"],
        "axes.titlesize": style["base_font_size"] + 1,
        "axes.labelsize": style["base_font_size"],
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "xtick.labelsize": style["base_font_size"] - 1,
        "ytick.labelsize": style["base_font_size"] - 1,
    })
    return style


def new_figure(style: Mapping[str, Any], *, width: float | None = None, height: float | None = None):
    return plt.subplots(
        figsize=(width or style["width_in"], height or style["height_in"])
    )


def save_fig(fig, path: Path, *, formats: Sequence[str] = ("png",), dpi: int | None = None, close: bool = True) -> list[Path]:
    """Write a figure to one or more formats and close it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for fmt in formats:
        out = path.with_suffix(f".{fmt.lstrip('.')}")
        fig.savefig(out, dpi=dpi or plt.rcParams["savefig.dpi"], format=fmt.lstrip("."))
        written.append(out)
    if close:
        plt.close(fig)
    return written


# --------------------------------------------------------------------------
# chart builders
# --------------------------------------------------------------------------


def fig_heatmap(
    matrix: np.ndarray,
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    *,
    style: Mapping[str, Any],
    title: str = "",
    cbar_label: str = "",
    cmap: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    annotate: bool | None = None,
):
    """Labelled heatmap. Annotates cells only when the grid is small enough to read."""
    data = np.asarray(matrix, dtype=float)
    if data.shape != (len(row_labels), len(col_labels)):
        raise ValueError(
            f"matrix shape {data.shape} does not match labels "
            f"({len(row_labels)}, {len(col_labels)})"
        )

    height = max(style["height_in"], 0.32 * len(row_labels) + 1.8)
    width = max(style["width_in"], 0.42 * len(col_labels) + 2.6)
    fig, ax = plt.subplots(figsize=(width, height))

    image = ax.imshow(
        data, cmap=cmap or style["heatmap_cmap"], vmin=vmin, vmax=vmax, aspect="auto"
    )
    ax.set_xticks(range(len(col_labels)), col_labels, rotation=45, ha="right")
    ax.set_yticks(range(len(row_labels)), row_labels)
    ax.grid(False)
    if title:
        ax.set_title(title)

    bar = fig.colorbar(image, ax=ax, shrink=0.85)
    if cbar_label:
        bar.set_label(cbar_label)

    should_annotate = (
        annotate if annotate is not None else data.size <= style["annotate_max_cells"]
    )
    if should_annotate:
        finite = data[np.isfinite(data)]
        midpoint = (finite.max() + finite.min()) / 2 if finite.size else 0.0
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                value = data[i, j]
                if not np.isfinite(value):
                    continue
                ax.text(
                    j, i, f"{value:.2f}",
                    ha="center", va="center", fontsize=style["base_font_size"] - 3,
                    color="white" if value < midpoint else "black",
                )
    return fig, ax


def fig_grouped_bar(
    categories: Sequence[str],
    series: Mapping[str, Sequence[float]],
    *,
    style: Mapping[str, Any],
    title: str = "",
    ylabel: str = "",
    colors: Mapping[str, str] | None = None,
    errors: Mapping[str, Sequence[float]] | None = None,
    horizontal: bool = False,
):
    """Grouped bars - the A-vs-B comparison shape used throughout the paper."""
    n_groups = len(categories)
    n_series = len(series)
    if n_series == 0:
        raise ValueError("no series to plot")

    span = 0.8 / n_series
    positions = np.arange(n_groups)
    length = max(style["height_in"], 0.3 * n_groups + 1.6)
    fig, ax = plt.subplots(
        figsize=(style["width_in"], length) if horizontal else (max(style["width_in"], 0.5 * n_groups + 2), style["height_in"])
    )

    for i, (name, values) in enumerate(series.items()):
        if len(values) != n_groups:
            raise ValueError(f"series {name!r} has {len(values)} values, expected {n_groups}")
        offset = (i - (n_series - 1) / 2) * span
        err = list(errors[name]) if errors and name in errors else None
        color = colors.get(name) if colors else None
        if horizontal:
            ax.barh(positions + offset, values, height=span, label=name, color=color, xerr=err, capsize=2)
        else:
            ax.bar(positions + offset, values, width=span, label=name, color=color, yerr=err, capsize=2)

    if horizontal:
        ax.set_yticks(positions, categories)
        ax.set_xlabel(ylabel)
        ax.invert_yaxis()
    else:
        ax.set_xticks(positions, categories, rotation=45, ha="right")
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if n_series > 1:
        ax.legend()
    return fig, ax


def fig_stacked_bar(
    categories: Sequence[str],
    layers: Mapping[str, Sequence[float]],
    *,
    style: Mapping[str, Any],
    title: str = "",
    ylabel: str = "proportion",
    colors: Mapping[str, str] | None = None,
):
    """Stacked composition bars - the emotion-distribution figure Phase 3 asks for."""
    positions = np.arange(len(categories))
    bottom = np.zeros(len(categories), dtype=float)
    fig, ax = plt.subplots(
        figsize=(max(style["width_in"], 0.5 * len(categories) + 2), style["height_in"])
    )
    for name, values in layers.items():
        values = np.asarray(values, dtype=float)
        if len(values) != len(categories):
            raise ValueError(f"layer {name!r} has {len(values)} values, expected {len(categories)}")
        ax.bar(
            positions, values, bottom=bottom, label=name,
            color=colors.get(name) if colors else None, width=0.75,
        )
        bottom += values
    ax.set_xticks(positions, categories, rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.01, 1.0), loc="upper left")
    return fig, ax


def fig_scatter(
    x: Sequence[float],
    y: Sequence[float],
    labels: Sequence[str] | None = None,
    groups: Sequence[str] | None = None,
    *,
    style: Mapping[str, Any],
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    colors: Mapping[str, str] | None = None,
    annotate: bool = True,
):
    """2D projection scatter (PCA / t-SNE) with optional point labels."""
    xs = np.asarray(x, dtype=float)
    ys = np.asarray(y, dtype=float)
    fig, ax = plt.subplots(figsize=(style["width_in"], style["height_in"]))

    if groups is not None:
        for name in dict.fromkeys(groups):
            mask = np.array([g == name for g in groups])
            ax.scatter(
                xs[mask], ys[mask], label=name, s=42, alpha=0.85,
                color=colors.get(name) if colors else None, edgecolors="none",
            )
        ax.legend()
    else:
        ax.scatter(xs, ys, s=42, alpha=0.85, edgecolors="none")

    if annotate and labels is not None:
        for xi, yi, text in zip(xs, ys, labels):
            ax.annotate(
                str(text), (xi, yi), fontsize=style["base_font_size"] - 3,
                xytext=(3, 3), textcoords="offset points", alpha=0.8,
            )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    return fig, ax


def fig_hist(
    series: Mapping[str, Sequence[float]],
    *,
    style: Mapping[str, Any],
    bins: int = 30,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "density",
    colors: Mapping[str, str] | None = None,
    density: bool = True,
):
    """Overlaid histograms - density by default so unequal corpus sizes compare fairly."""
    fig, ax = plt.subplots(figsize=(style["width_in"], style["height_in"]))
    for name, values in series.items():
        data = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
        if data.size == 0:
            continue
        ax.hist(
            data, bins=bins, alpha=0.55, label=name, density=density,
            color=colors.get(name) if colors else None,
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend()
    return fig, ax


def _drawable_subgraph(graph, *, max_nodes: int = 70, max_edges: int = 110):
    """Keep a readable backbone: drop isolates, cap nodes/edges, keep large components."""
    import networkx as nx

    g = graph.copy()
    g.remove_nodes_from(list(nx.isolates(g)))
    if g.number_of_nodes() == 0:
        return graph.copy()

    if g.number_of_nodes() > max_nodes:
        keep = {n for n, _ in sorted(g.degree(weight="weight"), key=lambda kv: -kv[1])[:max_nodes]}
        g = g.subgraph(keep).copy()
        g.remove_nodes_from(list(nx.isolates(g)))

    if g.number_of_edges() > max_edges:
        strongest = sorted(
            g.edges(data=True),
            key=lambda e: -float(e[2].get("weight", 1.0)),
        )[:max_edges]
        trimmed = nx.Graph()
        trimmed.add_nodes_from(g.nodes(data=True))
        trimmed.add_edges_from(strongest)
        trimmed.remove_nodes_from(list(nx.isolates(trimmed)))
        g = trimmed

    if g.number_of_nodes() and not nx.is_connected(g):
        components = sorted(nx.connected_components(g), key=len, reverse=True)
        keep: set = set()
        for comp in components:
            if len(comp) >= 3 and len(keep) + len(comp) <= max_nodes + 10:
                keep |= comp
            if len(keep) >= 20:
                break
        if not keep:
            keep = components[0]
        g = g.subgraph(keep).copy()
    return g


def _layout_components(graph, *, seed: int, iterations: int, prefer_kamada: bool) -> dict:
    """Layout each connected component, then tile them so clusters do not collapse
    to opposite corners of a mostly-empty canvas."""
    import networkx as nx

    components = [graph.subgraph(c).copy() for c in nx.connected_components(graph)]
    components.sort(key=lambda g: -g.number_of_nodes())
    n_comp = max(len(components), 1)
    cols = 1 if n_comp == 1 else 2
    rows = int(np.ceil(n_comp / cols))
    positions: dict = {}
    for i, comp in enumerate(components):
        n = comp.number_of_nodes()
        if n == 1:
            local = {next(iter(comp.nodes())): np.array([0.0, 0.0])}
        elif prefer_kamada or (n <= 25 and nx.is_connected(comp)):
            try:
                local = nx.kamada_kawai_layout(comp)
            except Exception:
                local = nx.spring_layout(
                    comp, seed=seed + i, iterations=iterations,
                    k=2.4 / np.sqrt(n), weight=None,
                )
        else:
            local = nx.spring_layout(
                comp, seed=seed + i, iterations=iterations,
                k=2.8 / np.sqrt(n), weight=None,
            )
        xs = np.array([p[0] for p in local.values()])
        ys = np.array([p[1] for p in local.values()])
        span_x = max(float(xs.max() - xs.min()), 1e-6)
        span_y = max(float(ys.max() - ys.min()), 1e-6)
        xs = (xs - xs.mean()) / span_x
        ys = (ys - ys.mean()) / span_y
        col, row = i % cols, i // cols
        ox = (col - (cols - 1) / 2.0) * 1.55
        oy = ((rows - 1) / 2.0 - row) * 1.55
        nodes = list(local.keys())
        for j, node in enumerate(nodes):
            positions[node] = np.array([xs[j] + ox, ys[j] + oy])
    return positions


def fig_network(
    graph,
    *,
    style: Mapping[str, Any],
    title: str = "",
    node_colors: Mapping[str, str] | None = None,
    node_sizes: Mapping[str, float] | None = None,
    layout: str = "spring",
    layout_seed: int = 7,
    layout_iterations: int = 400,
    label_top_n: int = 36,
    color_legend: Mapping[str, str] | None = None,
    max_nodes: int = 70,
    max_edges: int = 110,
):
    """Co-occurrence network. The layout seed is required, not optional -
    an unseeded spring layout redraws differently on every run.

    Dense graphs are thinned to a readable backbone (top-degree nodes and
    strongest edges). Isolates are dropped so spring layout cannot fling
    them to the corners. Labels sit on a white halo, offset from the hull.
    """
    import networkx as nx
    from matplotlib.patches import Patch

    if graph.number_of_nodes() == 0:
        raise ValueError("cannot draw an empty graph")

    drawn = _drawable_subgraph(graph, max_nodes=max_nodes, max_edges=max_edges)
    if drawn.number_of_nodes() == 0:
        drawn = graph.copy()

    n = drawn.number_of_nodes()
    if layout == "circular":
        positions = nx.circular_layout(drawn)
    else:
        positions = _layout_components(
            drawn,
            seed=layout_seed,
            iterations=max(layout_iterations, 400),
            prefer_kamada=(layout == "kamada_kawai"),
        )

    fig, ax = plt.subplots(figsize=(10.5, 10.5))
    ax.set_axis_off()
    ax.grid(False)

    weights = [float(drawn[u][v].get("weight", 1.0)) for u, v in drawn.edges()]
    widths = np.asarray(weights, dtype=float)
    if widths.size and widths.max() > widths.min():
        widths = 0.4 + 2.0 * (widths - widths.min()) / (widths.max() - widths.min())
    else:
        widths = np.full(max(widths.size, 1), 0.7)

    nx.draw_networkx_edges(
        drawn, positions, ax=ax, width=widths.tolist(),
        alpha=0.35, edge_color="#9AA0A6",
    )

    default_color = "#4C78A8"
    colors = [node_colors.get(n, default_color) if node_colors else default_color for n in drawn.nodes()]
    raw_sizes = np.array(
        [float(node_sizes.get(n, 120.0)) if node_sizes else 120.0 for n in drawn.nodes()],
        dtype=float,
    )
    if raw_sizes.max() > raw_sizes.min():
        sizes = 140 + 420 * (raw_sizes - raw_sizes.min()) / (raw_sizes.max() - raw_sizes.min())
    else:
        sizes = np.full(raw_sizes.size, 220.0)

    nx.draw_networkx_nodes(
        drawn, positions, ax=ax,
        node_color=colors,
        node_size=sizes.tolist(),
        linewidths=0.6,
        edgecolors="white",
        alpha=0.95,
    )

    ranked = sorted(drawn.degree(weight="weight"), key=lambda kv: -kv[1])[: min(label_top_n, n)]
    xs = np.array([positions[n][0] for n in drawn.nodes()])
    ys = np.array([positions[n][1] for n in drawn.nodes()])
    cx, cy = float(xs.mean()), float(ys.mean())
    font_size = max(7, int(style["base_font_size"]) - 2)
    used: list[tuple[float, float]] = []
    for node, _ in ranked:
        x, y = positions[node]
        dx, dy = x - cx, y - cy
        hypot = float(np.hypot(dx, dy)) or 1.0
        ox, oy = 10.0 * dx / hypot, 10.0 * dy / hypot
        # nudge away from already-placed labels
        for ux, uy in used:
            if abs((x + ox / 80.0) - ux) < 0.045 and abs((y + oy / 80.0) - uy) < 0.045:
                ox *= 1.55
                oy *= 1.55
        ax.annotate(
            str(node),
            (x, y),
            xytext=(ox, oy),
            textcoords="offset points",
            fontsize=font_size,
            ha="center",
            va="center",
            color="#1a1a1a",
            bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": "#dddddd", "alpha": 0.92},
        )
        used.append((x + ox / 80.0, y + oy / 80.0))

    if color_legend:
        handles = [Patch(facecolor=c, edgecolor="none", label=lab) for lab, c in color_legend.items()]
        ax.legend(handles=handles, loc="lower left", frameon=True, framealpha=0.9, fontsize=font_size)

    note = (
        f"{drawn.number_of_nodes()} nodes, {drawn.number_of_edges()} strongest edges "
        f"(isolates dropped; labels = highest-degree terms)"
    )
    ax.text(0.0, -0.02, note, transform=ax.transAxes, ha="left", va="top", fontsize=8, color="#555555")

    if title:
        ax.set_title(title, pad=12)
    ax.margins(0.08)
    fig.tight_layout()
    return fig, ax
