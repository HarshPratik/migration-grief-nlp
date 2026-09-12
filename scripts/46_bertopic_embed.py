"""Step 46a - precompute MiniLM embeddings for the full corpora (GPU).

BERTopic runs in the isolated .venv-bertopic environment (CPU torch); this
script runs under .venv-analysis where CUDA is available and encodes each
post once with all-MiniLM-L6-v2 (the same encoder the pipeline already uses
for MiniLM topics in 33_phase4_full_compare.py and 42_topics_over_time.py).

Outputs (data/analysis/vectors/):
    minilm_full_{a,b}.npy        - (n_posts, 384) float32 embeddings
    minilm_full_{a,b}_ids.json   - post_id order matching the npy rows

Example:
    .venv-analysis\\Scripts\\python.exe scripts/46_bertopic_embed.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analysis_config import load_analysis_config  # noqa: E402
from analysis_io import banner, item  # noqa: E402

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def main() -> int:
    from sentence_transformers import SentenceTransformer

    cfg = load_analysis_config()
    analysis_dir = cfg.path("analysis_dir")
    vectors_dir = analysis_dir / "vectors"
    vectors_dir.mkdir(parents=True, exist_ok=True)

    model = SentenceTransformer(MODEL_NAME)  # picks up CUDA automatically
    banner("MiniLM encoding (full corpora)")

    for label in ("a", "b"):
        corpus = pd.read_parquet(
            analysis_dir / f"corpus_{label}.parquet", columns=["post_id", "text"]
        )
        texts = corpus["text"].fillna("").astype(str).tolist()
        embeddings = model.encode(
            texts, batch_size=256, show_progress_bar=True,
            convert_to_numpy=True, normalize_embeddings=True,
        ).astype(np.float32)
        np.save(vectors_dir / f"minilm_full_{label}.npy", embeddings)
        (vectors_dir / f"minilm_full_{label}_ids.json").write_text(
            json.dumps(corpus["post_id"].astype(str).tolist()), encoding="utf-8"
        )
        item(f"corpus {label}: {embeddings.shape} -> minilm_full_{label}.npy")

    banner("Done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
