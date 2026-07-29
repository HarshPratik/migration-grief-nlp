"""Phase 2 - side-by-side PPMI vs contextual (BERT) comparison artifacts.

Loads reliability and semantic comparison tables from both backends when
present, joins on (unit, target), and writes agreement tables plus a simple
cos* scatter for domains.

Does not re-run inference. Safe to call after both pipelines have produced
outputs (or with only one backend - missing side is reported, partial joins
still written when possible).

Example:
    ...\\python.exe scripts/23_methods_compare.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analysis_config import load_analysis_config  # noqa: E402
from analysis_io import banner, item, out_path, save_summary, save_table  # noqa: E402
from plotting import apply_style, fig_scatter, save_fig  # noqa: E402


def _try_read(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_csv(path)
    return None


def _reliability_paths() -> dict[str, list[Path]]:
    paths: dict[str, list[Path]] = {"ppmi": [], "contextual": []}
    for unit in ("domain", "anchor"):
        paths["ppmi"].append(out_path("tables", "p2", 13, "reliability", unit))
        paths["contextual"].append(
            out_path("tables", "p2", 13, "reliability", unit, "contextual")
        )
    return paths


def _semantic_paths() -> dict[str, list[Path]]:
    paths: dict[str, list[Path]] = {"ppmi": [], "contextual": []}
    for unit in ("domain", "anchor"):
        paths["ppmi"].append(out_path("tables", "p2", 14, "semantic_comparison", unit))
        paths["contextual"].append(
            out_path("tables", "p2", 14, "semantic_comparison", unit, "contextual")
        )
    return paths


def _load_stacked(path_list: list[Path]) -> pd.DataFrame | None:
    frames = []
    for path in path_list:
        df = _try_read(path)
        if df is not None and len(df):
            frames.append(df)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare PPMI vs contextual method outputs")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = load_analysis_config(seed=args.seed)
    banner("Methods compare: PPMI vs contextual")

    rel_ppmi = _load_stacked(_reliability_paths()["ppmi"])
    rel_ctx = _load_stacked(_reliability_paths()["contextual"])
    sem_ppmi = _load_stacked(_semantic_paths()["ppmi"])
    sem_ctx = _load_stacked(_semantic_paths()["contextual"])

    summary: dict = {
        "found": {
            "reliability_ppmi": rel_ppmi is not None,
            "reliability_contextual": rel_ctx is not None,
            "semantic_ppmi": sem_ppmi is not None,
            "semantic_contextual": sem_ctx is not None,
        }
    }

    for name, present in summary["found"].items():
        item(f"{name}: {'yes' if present else 'MISSING'}")

    # ---- reliability join -------------------------------------------------
    if rel_ppmi is not None and rel_ctx is not None:
        left = rel_ppmi.copy()
        right = rel_ctx.copy()
        # Disambiguate overlapping columns
        keep_keys = ["unit", "target"]
        left_cols = keep_keys + [
            c for c in left.columns
            if c not in keep_keys and c != "backend"
        ]
        right_cols = keep_keys + [
            c for c in right.columns
            if c not in keep_keys and c != "backend"
        ]
        left = left[left_cols].rename(
            columns={c: f"{c}_ppmi" for c in left_cols if c not in keep_keys}
        )
        right = right[right_cols].rename(
            columns={c: f"{c}_contextual" for c in right_cols if c not in keep_keys}
        )
        rel_join = left.merge(right, on=keep_keys, how="outer")
        save_table(
            rel_join,
            out_path("tables", "p2", 23, "methods_reliability_compare"),
            parquet=False,
        )
        summary["n_reliability_rows"] = int(len(rel_join))
        item(f"reliability compare rows: {len(rel_join)}")
    else:
        item("skipping reliability join (need both backends)")
        summary["n_reliability_rows"] = 0

    # ---- semantic join + agreement ----------------------------------------
    if sem_ppmi is not None and sem_ctx is not None:
        left = sem_ppmi.copy()
        right = sem_ctx.copy()
        keep_keys = ["unit", "target"]
        metric_cols = [
            "cosine_raw", "cos_star", "rbo", "jaccard_at_k", "perm_p", "fdr_q",
            "significant", "in_fdr_family", "reliability_a", "reliability_b",
            "reliability_ceiling", "f_a", "f_b",
        ]
        left_keep = keep_keys + [c for c in metric_cols if c in left.columns]
        right_keep = keep_keys + [c for c in metric_cols if c in right.columns]
        left = left[left_keep].rename(
            columns={c: f"{c}_ppmi" for c in left_keep if c not in keep_keys}
        )
        right = right[right_keep].rename(
            columns={c: f"{c}_contextual" for c in right_keep if c not in keep_keys}
        )
        sem_join = left.merge(right, on=keep_keys, how="outer")
        save_table(
            sem_join,
            out_path("tables", "p2", 23, "methods_semantic_compare"),
            parquet=False,
        )
        summary["n_semantic_rows"] = int(len(sem_join))
        item(f"semantic compare rows: {len(sem_join)}")

        # Agreement on significance call
        if "significant_ppmi" in sem_join.columns and "significant_contextual" in sem_join.columns:
            def _sig_label(row) -> str:
                a = row.get("significant_ppmi")
                b = row.get("significant_contextual")
                if pd.isna(a) or pd.isna(b):
                    return "missing"
                a_bool = bool(a)
                b_bool = bool(b)
                if a_bool and b_bool:
                    return "both_yes"
                if (not a_bool) and (not b_bool):
                    return "both_no"
                return "disagree"

            agree = sem_join[keep_keys].copy()
            for col in (
                "significant_ppmi", "significant_contextual",
                "cos_star_ppmi", "cos_star_contextual",
                "perm_p_ppmi", "perm_p_contextual",
                "in_fdr_family_ppmi", "in_fdr_family_contextual",
            ):
                if col in sem_join.columns:
                    agree[col] = sem_join[col]
            agree["agreement"] = sem_join.apply(_sig_label, axis=1)
            save_table(
                agree,
                out_path("tables", "p2", 23, "methods_agreement"),
                parquet=False,
            )
            counts = agree["agreement"].value_counts().to_dict()
            summary["agreement_counts"] = {str(k): int(v) for k, v in counts.items()}
            item(f"significance agreement: {summary['agreement_counts']}")
        else:
            item("significant columns missing; agreement table skipped")
    else:
        item("skipping semantic join (need both backends)")
        summary["n_semantic_rows"] = 0

    # ---- scatter figure: domain cos* --------------------------------------
    figure_paths: list[str] = []
    if sem_ppmi is not None and sem_ctx is not None:
        d_ppmi = sem_ppmi[sem_ppmi["unit"] == "domain"] if "unit" in sem_ppmi.columns else sem_ppmi
        d_ctx = sem_ctx[sem_ctx["unit"] == "domain"] if "unit" in sem_ctx.columns else sem_ctx
        if "cos_star" in d_ppmi.columns and "cos_star" in d_ctx.columns:
            merged = d_ppmi[["target", "cos_star"]].rename(columns={"cos_star": "cos_star_ppmi"}).merge(
                d_ctx[["target", "cos_star"]].rename(columns={"cos_star": "cos_star_contextual"}),
                on="target",
                how="inner",
            )
            if len(merged) > 0:
                style = apply_style(cfg.section("figures"))
                xs = merged["cos_star_ppmi"].to_numpy(dtype=float)
                ys = merged["cos_star_contextual"].to_numpy(dtype=float)
                labels = merged["target"].astype(str).tolist()
                fig, ax = fig_scatter(
                    xs, ys, labels, None,
                    style=style,
                    title="Domain cos*: PPMI vs contextual",
                    xlabel="cos* (PPMI)",
                    ylabel="cos* (contextual / BERT)",
                )
                # y=x reference
                lo = float(np.nanmin([xs.min(), ys.min(), 0.0]))
                hi = float(np.nanmax([xs.max(), ys.max(), 1.0]))
                ax.plot([lo, hi], [lo, hi], linestyle="--", color="#888888", linewidth=1, alpha=0.7)
                written = save_fig(
                    fig,
                    out_path("figures", "p2", 23, "cosstar_ppmi_vs_contextual", ext="png"),
                    formats=list(cfg.get("figures.formats")),
                )
                # Also ensure the canonical filename requested in the brief.
                canonical = ROOT / "outputs" / "figures" / "p2_23_cosstar_ppmi_vs_contextual.png"
                if written and written[0] != canonical and written[0].exists():
                    # out_path already produces p2_23_cosstar_ppmi_vs_contextual.png
                    pass
                figure_paths = [str(p) for p in written]
                item(f"wrote figure(s): {figure_paths}")
            else:
                item("no overlapping domain targets for scatter")
        else:
            item("cos_star missing; scatter skipped")

    summary["figures"] = figure_paths
    save_summary(summary, out_path("summaries", "p2", 23, "methods_compare", ext="json"))
    banner("Done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
