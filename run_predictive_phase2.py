"""Run empirical Word2Vec + FastText evaluation, then refresh methods tables.

These models are trained on the research corpora (unlike frozen BERT). The
headline check is split-half reliability of second-order profiles and neighbour
lists — so the paper can show measured performance at the current sample size,
not an a priori dismissal.

Example:
    python run_predictive_phase2.py
    python run_predictive_phase2.py -- --n-splits 15 --n-bootstrap 30 --n-permutations 40
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description="Word2Vec/FastText empirical evaluation")
    parser.add_argument(
        "extra",
        nargs="*",
        help="flags forwarded to 25_predictive_eval.py (after --)",
    )
    args = parser.parse_args()

    steps = [
        (
            "predictive-eval",
            "scripts/25_predictive_eval.py",
            ["--algorithms", "word2vec", "fasttext", "--unit", "both", *args.extra],
        ),
        (
            "methods-dashboard",
            "scripts/23b_methods_dashboard.py",
            [],
        ),
    ]
    for name, relative, step_args in steps:
        script = ROOT / relative
        if name == "methods-dashboard" and not script.exists():
            # Fallback: original two-way compare still useful
            script = ROOT / "scripts" / "23_methods_compare.py"
            step_args = []
        print(f"\n=== RUN {name}: {script.name} {' '.join(step_args)} ===", flush=True)
        result = subprocess.run(
            [sys.executable, str(script), *step_args],
            cwd=ROOT,
            check=False,
        )
        if result.returncode != 0:
            print(f"FAILED: {name} exited {result.returncode}", file=sys.stderr)
            return result.returncode
    print("\n=== PREDICTIVE PHASE 2 COMPLETE ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
