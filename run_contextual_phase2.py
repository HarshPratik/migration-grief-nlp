"""Run the Phase 2 contextual (BERT) path and methods comparison.

PPMI remains the default in run_analysis.py so existing outputs stay intact.
This runner is additive:

  1. 12_build_embeddings.py --backend contextual --unit both
  2. 13_reliability_gate.py --backend contextual
  3. 14_compare_semantics.py --backend contextual --unit both
  4. 18_semantic_figures.py --backend contextual
  5. 23_methods_compare.py   (joins PPMI + contextual tables if both present)

Contextual defaults for permutations/bootstrap/splits are applied inside the
scripts (min(config, 500) / min(config, 25)); pass extra args after -- to
forward flags, e.g.:

    python run_contextual_phase2.py -- --n-permutations 200

Requires torch + transformers (see requirements-analysis-torch.txt).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

STEPS = (
    ("vectors", "scripts/12_build_embeddings.py", ("--backend", "contextual", "--unit", "both")),
    ("reliability", "scripts/13_reliability_gate.py", ("--backend", "contextual")),
    ("semantics", "scripts/14_compare_semantics.py", ("--backend", "contextual", "--unit", "both")),
    ("semantic-figures", "scripts/18_semantic_figures.py", ("--backend", "contextual")),
    ("methods-compare", "scripts/23_methods_compare.py", ()),
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Phase 2 contextual (BERT) pipeline")
    parser.add_argument(
        "--from-step",
        choices=[name for name, _, _ in STEPS],
        default="vectors",
    )
    parser.add_argument(
        "--through-step",
        choices=[name for name, _, _ in STEPS],
        default="methods-compare",
    )
    parser.add_argument(
        "extra",
        nargs="*",
        help="extra CLI flags forwarded to each step (use after --)",
    )
    args = parser.parse_args()

    names = [name for name, _, _ in STEPS]
    start = names.index(args.from_step)
    stop = names.index(args.through_step)
    if start > stop:
        parser.error("--from-step must not come after --through-step")

    for name, relative, fixed in STEPS[start : stop + 1]:
        script = ROOT / relative
        # Forward extras only to steps that accept them (not 23).
        step_args = list(fixed)
        if name in ("reliability", "semantics", "vectors", "semantic-figures") and args.extra:
            step_args.extend(args.extra)
        print(f"\n=== RUN {name}: {relative} {' '.join(step_args)} ===", flush=True)
        result = subprocess.run(
            [sys.executable, str(script), *step_args],
            cwd=ROOT,
            check=False,
        )
        if result.returncode != 0:
            print(f"FAILED: {name} exited {result.returncode}", file=sys.stderr)
            return result.returncode

    print("\n=== CONTEXTUAL PHASE 2 COMPLETE ===")
    print("PPMI outputs were not modified. Methods compare is under outputs/tables/p2_23_*.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
