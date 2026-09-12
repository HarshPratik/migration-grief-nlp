"""Phase 3 follow-up - figures for the TRUE ABSA run (script 45).

Stacked polarity bars per domain (A vs B) and an A-minus-B difference heatmap,
mirroring p3_19_* but for aspect-directed polarity instead of window emotion.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analysis_config import load_analysis_config  # noqa: E402
from analysis_io import banner, item, out_path, save_summary  # noqa: E402
from plotting import apply_style, fig_heatmap, fig_stacked_bar, save_fig  # noqa: E402

POLARITY_COLORS = {"negative": "#D4575B", "neutral": "#9DB2C8", "positive": "#59A14F"}


def main() -> int:
    cfg = load_analysis_config()
    table_path = out_path("tables", "p3", 45, "absa_comparison")
    if not table_path.exists():
        raise SystemExit(f"missing {table_path}; run scripts/45_absa_true.py first")
    table = pd.read_csv(table_path)
    domains = table[table["level"] == "domain"].set_index("target")
    if domains.empty:
        raise SystemExit("no domain rows in p3_45_absa_comparison.csv")

    polarities = [c[2:] for c in domains.columns if c.startswith("a_") and not c.startswith("a_a")]
    order = [p for p in ("negative", "neutral", "positive") if p in polarities] + [
        p for p in polarities if p not in ("negative", "neutral", "positive")
    ]
    style = apply_style(cfg.section("figures"))
    labels = cfg.get("figures.corpus_labels") or {"a": "Corpus A (migration)", "b": "Corpus B (grief)"}

    # ---- stacked bars: one panel of categories per corpus ------------------
    for corpus in ("a", "b"):
        layers = {p: domains[f"{corpus}_{p}"].tolist() for p in order}
        fig, _ = fig_stacked_bar(
            list(domains.index), layers, style=style,
            title=f"Aspect-directed polarity by seed domain - {labels.get(corpus, corpus)}",
            ylabel="mean probability",
            colors={p: POLARITY_COLORS.get(p) for p in order},
        )
        written = save_fig(fig, out_path("figures", "p3", 45, f"absa_stacked_domain__{corpus}", ext="png"))
        item(f"wrote {written[0].name}")

    # ---- A-minus-B difference heatmap --------------------------------------
    diff = np.array([[row[f"a_{p}"] - row[f"b_{p}"] for p in order] for _, row in domains.iterrows()])
    fig, _ = fig_heatmap(
        diff, list(domains.index), order, style=style,
        title="ABSA polarity difference (migration - grief), by seed domain",
        cbar_label="A - B mean probability", cmap="RdBu_r",
        vmin=-np.abs(diff).max(), vmax=np.abs(diff).max(), annotate=True,
    )
    written = save_fig(fig, out_path("figures", "p3", 45, "absa_difference_heatmap", ext="png"))
    item(f"wrote {written[0].name}")

    save_summary(
        {"source_table": table_path.name, "domains": list(domains.index), "polarities": order},
        out_path("summaries", "p3", 45, "absa_figures", ext="json"),
    )
    banner("Done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
