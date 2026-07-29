"""Multi-method Phase 2 dashboard: PPMI, BERT, Word2Vec, FastText.

Stacks reliability and semantic tables from every backend present on disk and
writes a single comparison-friendly summary for the paper methods section.

Example:
    python scripts/23b_methods_dashboard.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analysis_io import banner, item, out_path, save_summary, save_table  # noqa: E402

BACKENDS = ("ppmi", "contextual", "word2vec", "fasttext")


def _rel_path(unit: str, backend: str) -> Path:
    if backend == "ppmi":
        return out_path("tables", "p2", 13, "reliability", unit)
    return out_path("tables", "p2", 13, "reliability", unit, backend)


def _sem_path(unit: str, backend: str) -> Path:
    if backend == "ppmi":
        return out_path("tables", "p2", 14, "semantic_comparison", unit)
    return out_path("tables", "p2", 14, "semantic_comparison", unit, backend)


def _load_all(kind: str) -> pd.DataFrame | None:
    frames = []
    for backend in BACKENDS:
        for unit in ("domain", "anchor"):
            path = _rel_path(unit, backend) if kind == "reliability" else _sem_path(unit, backend)
            if not path.exists():
                continue
            df = pd.read_csv(path)
            if "backend" not in df.columns:
                df["backend"] = backend
            if "unit" not in df.columns:
                df["unit"] = unit
            frames.append(df)
            item(f"loaded {path.name} ({len(df)} rows)")
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Multi-method methods dashboard")
    parser.parse_args()

    banner("Multi-method dashboard (PPMI / BERT / Word2Vec / FastText)")
    rel = _load_all("reliability")
    sem = _load_all("semantic")

    summary: dict = {"backends_reliability": [], "backends_semantic": []}

    if rel is not None:
        save_table(rel, out_path("tables", "p2", 23, "all_methods_reliability"), parquet=False)
        summary["backends_reliability"] = sorted(rel["backend"].astype(str).unique().tolist())
        # Domain-level reliability scoreboard
        domain = rel[rel["unit"] == "domain"].copy()
        if len(domain):
            score = (
                domain.groupby("backend", as_index=False)
                .agg(
                    n_targets=("target", "count"),
                    n_usable=("usable", "sum"),
                    mean_rel_a=("reliability_a", "mean"),
                    mean_rel_b=("reliability_b", "mean"),
                    mean_jac_a=("neighbour_jaccard_a", "mean"),
                    mean_jac_b=("neighbour_jaccard_b", "mean"),
                )
            )
            for col in ("mean_rel_a", "mean_rel_b", "mean_jac_a", "mean_jac_b"):
                if col in score.columns:
                    score[col] = score[col].round(4)
            save_table(
                score,
                out_path("tables", "p2", 23, "reliability_scoreboard_domain"),
                parquet=False,
            )
            item("domain reliability scoreboard:")
            for _, row in score.iterrows():
                item(
                    f"  {row['backend']:<12} usable={int(row['n_usable'])}/{int(row['n_targets'])}  "
                    f"relA={row['mean_rel_a']:.3f} relB={row['mean_rel_b']:.3f}  "
                    f"jacA={row['mean_jac_a']:.3f} jacB={row['mean_jac_b']:.3f}"
                )
            summary["domain_scoreboard"] = score.to_dict(orient="records")

        # Wide reliability for domains (easy paper table)
        if len(domain):
            wide = domain.pivot_table(
                index="target",
                columns="backend",
                values="min_reliability",
                aggfunc="first",
            ).reset_index()
            save_table(
                wide,
                out_path("tables", "p2", 23, "domain_min_reliability_by_method"),
                parquet=False,
            )
    else:
        item("no reliability tables found")

    if sem is not None:
        save_table(sem, out_path("tables", "p2", 23, "all_methods_semantic"), parquet=False)
        summary["backends_semantic"] = sorted(sem["backend"].astype(str).unique().tolist())
        domain = sem[sem["unit"] == "domain"].copy()
        if len(domain) and "cos_star" in domain.columns:
            wide = domain.pivot_table(
                index="target",
                columns="backend",
                values="cos_star",
                aggfunc="first",
            ).reset_index()
            save_table(
                wide,
                out_path("tables", "p2", 23, "domain_cosstar_by_method"),
                parquet=False,
            )
            item("domain cos* by method written")
    else:
        item("no semantic tables found")

    save_summary(summary, out_path("summaries", "p2", 23, "methods_dashboard", ext="json"))
    banner("Done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
