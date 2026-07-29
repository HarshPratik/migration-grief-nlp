"""Phase 2 helper - long-form top-k neighbor export from semantic comparison CSVs.

Reads every `p2_14_semantic_comparison*.csv` under outputs/tables/, expands the
`neighbours_a` / `neighbours_b` semicolon lists into one row per (target, corpus,
rank), and writes:

    outputs/tables/p2_14_neighbors_top15__ppmi.csv

Columns: backend, unit, target, corpus, rank, neighbor

Does not recompute embeddings; pure reshape of existing comparison tables.
Safe to re-run whenever new comparison CSVs appear (domain, anchor, or both).

Example:
    ...\\python.exe scripts/14b_export_neighbors.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analysis_io import OUTPUTS_DIR, banner, item, out_path, save_table  # noqa: E402


def _infer_backend(path: Path) -> str:
    """Map comparison CSV stem/facets to a backend label.

    Default PPMI tables are `p2_14_semantic_comparison__{unit}.csv`.
    Optional BERT/contextual tables may include `bert` / `contextual` in the name.
    """
    stem = path.stem.lower()
    for token in ("bert", "contextual", "ppmi", "svd", "glove"):
        if f"__{token}" in stem or stem.endswith(f"_{token}") or f"_{token}__" in stem:
            return token
    return "ppmi"


def expand_neighbors(df: pd.DataFrame, *, backend: str) -> pd.DataFrame:
    """Expand neighbours_a / neighbours_b into long form."""
    required = {"unit", "target", "neighbours_a", "neighbours_b"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"comparison table missing columns: {sorted(missing)}")

    rows: list[dict] = []
    for _, rec in df.iterrows():
        unit = str(rec["unit"])
        target = str(rec["target"])
        for corpus, col in (("a", "neighbours_a"), ("b", "neighbours_b")):
            raw = rec[col]
            if pd.isna(raw) or str(raw).strip() == "":
                continue
            neighbors = [n.strip() for n in str(raw).split(";") if n.strip()]
            for rank, neighbor in enumerate(neighbors, start=1):
                rows.append({
                    "backend": backend,
                    "unit": unit,
                    "target": target,
                    "corpus": corpus,
                    "rank": rank,
                    "neighbor": neighbor,
                })
    return pd.DataFrame(rows, columns=["backend", "unit", "target", "corpus", "rank", "neighbor"])


def main() -> int:
    parser = argparse.ArgumentParser(description="Long-form neighbor export from p2_14 comparison CSVs")
    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=None,
        help="Directory holding p2_14_semantic_comparison*.csv (default: outputs/tables)",
    )
    parser.add_argument(
        "--pattern",
        default="p2_14_semantic_comparison*.csv",
        help="Glob under tables-dir for comparison CSVs",
    )
    args = parser.parse_args()

    tables_dir = args.tables_dir or (OUTPUTS_DIR / "tables")
    paths = sorted(tables_dir.glob(args.pattern))
    # Skip already-long-form neighbor exports if the glob is too broad.
    paths = [p for p in paths if "neighbors" not in p.stem.lower()]

    banner("14b export neighbors")
    if not paths:
        raise SystemExit(f"no comparison CSVs matching {args.pattern!r} under {tables_dir}")

    frames: list[pd.DataFrame] = []
    for path in paths:
        backend = _infer_backend(path)
        table = pd.read_csv(path)
        long = expand_neighbors(table, backend=backend)
        item(f"{path.name}: {len(table)} targets -> {len(long)} neighbor rows (backend={backend})")
        frames.append(long)

    out = pd.concat(frames, ignore_index=True)
    # Prefer a stable order for diffs / hand inspection.
    out = out.sort_values(["backend", "unit", "target", "corpus", "rank"]).reset_index(drop=True)

    # One combined PPMI-facing artifact (user-requested name). If multiple
    # backends appear, write one file per backend plus the all-backend frame
    # only when a single backend is present under the canonical name.
    backends = sorted(out["backend"].unique())
    written: list[Path] = []
    for backend in backends:
        subset = out[out["backend"] == backend]
        dest = out_path("tables", "p2", 14, "neighbors_top15", backend)
        save_table(subset, dest, parquet=False)
        written.append(dest)
        item(f"wrote {dest.name}  rows={len(subset)}")

    banner("Done")
    item(f"backends: {', '.join(backends)}")
    item(f"total neighbor rows: {len(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
