"""Run the reproducible Phase 2-4 analysis pipeline in dependency order.

Default chain uses the PPMI vector backend (config vectors.backend: ppmi).
For the parallel BERT/contextual Phase 2 package (encode once, cache token
vectors, reliability + comparison + methods join), run:

    python run_contextual_phase2.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

STEPS = (
    ("environment", "scripts/check_env.py", ()),
    ("reprocess", "scripts/08_reprocess_tokens.py", ()),
    ("coverage", "scripts/09_anchor_coverage.py", ()),
    ("keyness", "scripts/10_keyness_prevalence.py", ()),
    ("windows", "scripts/11_extract_windows.py", ()),
    ("vectors", "scripts/12_build_embeddings.py", ()),  # PPMI by config default
    ("reliability", "scripts/13_reliability_gate.py", ()),
    ("semantics", "scripts/14_compare_semantics.py", ()),
    ("emotion", "scripts/15_emotion_absa.py", ()),
    ("network", "scripts/16_cooccurrence.py", ()),
    ("topics", "scripts/17_topic_model.py", ()),
    ("annotation-sample", "scripts/22_annotation_sample.py", ()),
    ("semantic-figures", "scripts/18_semantic_figures.py", ()),
    ("emotion-figures", "scripts/19_emotion_figures.py", ()),
    ("network-topic-figures", "scripts/20_network_topic_figures.py", ()),
    ("bundle", "scripts/21_export_bundle.py", ()),
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Phase 2-4 analysis")
    parser.add_argument("--from-step", choices=[name for name, _, _ in STEPS], default="environment")
    parser.add_argument("--through-step", choices=[name for name, _, _ in STEPS], default="bundle")
    parser.add_argument("--replace-bundle", action="store_true",
                        help="allow the final export step to replace an existing bundle")
    args = parser.parse_args()

    names = [name for name, _, _ in STEPS]
    start = names.index(args.from_step)
    stop = names.index(args.through_step)
    if start > stop:
        parser.error("--from-step must not come after --through-step")

    for name, relative, extra in STEPS[start : stop + 1]:
        script = ROOT / relative
        step_args = list(extra)
        if name == "bundle" and args.replace_bundle:
            step_args.append("--replace")
        print(f"\n=== RUN {name}: {relative} ===", flush=True)
        result = subprocess.run([sys.executable, str(script), *step_args], cwd=ROOT, check=False)
        if result.returncode != 0:
            print(f"FAILED: {name} exited {result.returncode}", file=sys.stderr)
            return result.returncode
    print("\n=== ANALYSIS COMPLETE ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
