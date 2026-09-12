"""Step 46b - real BERTopic per corpus (runs under .venv-bertopic).

Fulfils the literal Phase 4 / New.docx requirement that was previously
substituted: BERTopic (UMAP + HDBSCAN + c-TF-IDF) fitted independently on
Corpus A and Corpus B, with dynamic topic modelling via topics_over_time.

Inputs (produced under .venv-analysis):
    data/analysis/corpus_{a,b}.parquet        (Phase 1 frozen output is read-only input here)
    data/analysis/vectors/minilm_full_{a,b}.npy + _ids.json   (46_bertopic_embed.py)
    config/anchors.yaml                       (for the Digital-Absence mix audit)

Design:
  - precomputed MiniLM embeddings (normalized) -> BERTopic(embedding_model=None)
  - UMAP n_neighbors=15, n_components=5, min_dist=0.0, metric cosine, seeded
  - HDBSCAN min_cluster_size=100 (1% of a 10k corpus), prediction_data on
  - CountVectorizer: english stop words, min_df=10, unigrams+bigrams
  - nr_topics=15 for comparability with the independent LDA/NMF/MiniLM runs
    (BERTopic's own reduction; initial cluster count is reported too)

Outputs:
  outputs/tables/p4_46_bertopic_topics__{a,b}.csv          - top 10 words + c-TF-IDF weights per topic
  outputs/tables/p4_46_bertopic_doc_topics__{a,b}.csv      - post_id -> topic
  outputs/tables/p4_46_digital_absence_mix__bertopic.csv   - unforced mix audit (both corpora)
  outputs/tables/p5_46_bertopic_topics_over_time__{a,b}.csv
  outputs/figures/p4_46_interactive_topics__bertopic__{a,b}.html
  outputs/figures/p4_46_barchart__bertopic__{a,b}.html
  outputs/figures/p5_46_topics_over_time__bertopic__{a,b}.html
  outputs/summaries/p4_46_bertopic.json

Example:
    .venv-bertopic\\Scripts\\python.exe scripts/46_bertopic_run.py
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT / "data" / "analysis"
OUTPUTS_DIR = ROOT / "outputs"
ANCHORS_PATH = ROOT / "config" / "anchors.yaml"

SEED = 20260726
TOP_WORDS = 10


def load_anchor_domains() -> dict[str, str]:
    data = yaml.safe_load(ANCHORS_PATH.read_text(encoding="utf-8"))
    out: dict[str, str] = {}
    for domain_name, domain in data["domains"].items():
        for anchor in domain.get("anchors", []):
            term = anchor["term"] if isinstance(anchor, dict) else str(anchor)
            out[str(term).lower()] = domain_name
    return out


def topic_terms_table(model, corpus_label: str, topic_sizes: pd.DataFrame) -> pd.DataFrame:
    """Top-10 c-TF-IDF words + weights per topic (the table Phase 4 asks for)."""
    rows = []
    sizes = dict(zip(topic_sizes["Topic"], topic_sizes["Count"]))
    for topic_id in sorted(t for t in model.get_topics() if t != -1):
        for rank, (word, weight) in enumerate(model.get_topic(topic_id)[:TOP_WORDS], start=1):
            rows.append({
                "corpus": corpus_label, "topic": int(topic_id),
                "n_docs": int(sizes.get(topic_id, 0)),
                "rank": rank, "word": word, "weight": round(float(weight), 5),
            })
    return pd.DataFrame(rows)


def mix_audit(model, corpus_label: str, anchor_domain: dict[str, str]) -> list[dict]:
    """Unforced Digital-Absence check: does any topic's top-10 contain BOTH?"""
    rows = []
    for topic_id in sorted(t for t in model.get_topics() if t != -1):
        words = [w for w, _ in model.get_topic(topic_id)[:TOP_WORDS]]
        domains_hit = {anchor_domain[w.lower()] for w in words if w.lower() in anchor_domain}
        rows.append({
            "corpus": corpus_label, "topic": int(topic_id),
            "top_words": ", ".join(words),
            "anchor_domains_in_top10": ", ".join(sorted(domains_hit)) or "(none)",
            "has_digital": "digital" in domains_hit,
            "has_absence": "absence" in domains_hit,
            "digital_absence_mix": {"digital", "absence"} <= domains_hit,
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="BERTopic per corpus (real BERTopic run)")
    parser.add_argument("--corpus", choices=["a", "b", "both"], default="both")
    parser.add_argument("--min-cluster-size", type=int, default=10)
    parser.add_argument("--cluster-selection", choices=["eom", "leaf"], default="eom")
    parser.add_argument("--nr-topics", type=int, default=15)
    parser.add_argument("--time-bins", type=int, default=12)
    parser.add_argument("--skip-time", action="store_true")
    args = parser.parse_args()

    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP

    labels = ["a", "b"] if args.corpus == "both" else [args.corpus]
    anchor_domain = load_anchor_domains()
    (OUTPUTS_DIR / "tables").mkdir(parents=True, exist_ok=True)
    (OUTPUTS_DIR / "figures").mkdir(parents=True, exist_ok=True)
    (OUTPUTS_DIR / "summaries").mkdir(parents=True, exist_ok=True)

    summary: dict = {
        "model": "BERTopic (UMAP + HDBSCAN + c-TF-IDF)",
        "embeddings": "sentence-transformers/all-MiniLM-L6-v2, precomputed under .venv-analysis (CUDA)",
        "umap": {"n_neighbors": 15, "n_components": 5, "min_dist": 0.0, "metric": "cosine"},
        "hdbscan_min_cluster_size": args.min_cluster_size,
        "hdbscan_cluster_selection": args.cluster_selection,
        "nr_topics": args.nr_topics,
        "seed": SEED,
        "corpora": {},
    }
    mix_rows_all: list[dict] = []

    for label in labels:
        print(f"\n=== BERTopic: corpus {label} ===")
        corpus = pd.read_parquet(
            ANALYSIS_DIR / f"corpus_{label}.parquet",
            columns=["post_id", "text", "created_at"],
        )
        ids = json.loads(
            (ANALYSIS_DIR / "vectors" / f"minilm_full_{label}_ids.json").read_text(encoding="utf-8")
        )
        corpus = corpus.assign(post_id=corpus["post_id"].astype(str)).set_index("post_id").loc[ids]
        docs = corpus["text"].fillna("").astype(str).tolist()
        embeddings = np.load(ANALYSIS_DIR / "vectors" / f"minilm_full_{label}.npy")
        assert embeddings.shape[0] == len(docs), "embedding/doc order mismatch"

        umap_model = UMAP(
            n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=SEED
        )
        hdbscan_model = HDBSCAN(
            min_cluster_size=args.min_cluster_size,
            cluster_selection_method=args.cluster_selection,
            prediction_data=True,
        )
        # BERTopic fits this vectorizer on one aggregated document PER CLUSTER,
        # so min_df must stay well below the plausible cluster count.
        vectorizer = CountVectorizer(stop_words="english", min_df=2, ngram_range=(1, 2))

        model = BERTopic(
            embedding_model=None,  # embeddings are precomputed
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer,
            nr_topics=args.nr_topics,
            top_n_words=TOP_WORDS,
            verbose=True,
        )
        topics, _ = model.fit_transform(docs, embeddings=embeddings)

        n_initial = int(len(set(model.hdbscan_model.labels_)) - (1 if -1 in model.hdbscan_model.labels_ else 0))
        n_outliers_raw = int((np.asarray(topics) == -1).sum())

        # HDBSCAN marks genuinely off-manifold posts as -1. Reassign them to the
        # nearest topic centroid in embedding space (standard BERTopic practice)
        # so topic prevalence and topics_over_time cover the whole corpus.
        topics = model.reduce_outliers(docs, topics, strategy="embeddings", embeddings=embeddings)
        model.update_topics(docs, topics=topics, vectorizer_model=vectorizer)

        topic_sizes = model.get_topic_info()
        n_outliers = int((np.asarray(topics) == -1).sum())
        print(f"  initial HDBSCAN clusters: {n_initial}; after reduction: "
              f"{len([t for t in set(topics) if t != -1])}; "
              f"outliers: {n_outliers_raw} ({n_outliers_raw/len(docs):.1%}) raw -> "
              f"{n_outliers} after embedding-space reassignment")

        terms = topic_terms_table(model, label, topic_sizes)
        terms.to_csv(OUTPUTS_DIR / "tables" / f"p4_46_bertopic_topics__{label}.csv", index=False)

        doc_topics = pd.DataFrame({"post_id": ids, "topic": topics})
        doc_topics.to_csv(
            OUTPUTS_DIR / "tables" / f"p4_46_bertopic_doc_topics__{label}.csv", index=False
        )

        mix_rows_all.extend(mix_audit(model, label, anchor_domain))

        # Interactive views (Phase 4 brief: "interactive topic clusters")
        model.visualize_topics().write_html(
            OUTPUTS_DIR / "figures" / f"p4_46_interactive_topics__bertopic__{label}.html"
        )
        model.visualize_barchart(top_n_topics=args.nr_topics, n_words=8).write_html(
            OUTPUTS_DIR / "figures" / f"p4_46_barchart__bertopic__{label}.html"
        )

        # Dynamic topic modelling (New.docx Task 3, now with real BERTopic)
        if not args.skip_time:
            # BERTopic indexes timestamps positionally; pass a plain list.
            timestamps = list(pd.to_datetime(corpus["created_at"], utc=True))
            tot = model.topics_over_time(docs, timestamps, nr_bins=args.time_bins)
            tot.to_csv(
                OUTPUTS_DIR / "tables" / f"p5_46_bertopic_topics_over_time__{label}.csv", index=False
            )
            model.visualize_topics_over_time(tot, top_n_topics=args.nr_topics).write_html(
                OUTPUTS_DIR / "figures" / f"p5_46_topics_over_time__bertopic__{label}.html"
            )

        summary["corpora"][label] = {
            "n_docs": len(docs),
            "n_initial_clusters": n_initial,
            "n_topics_after_reduction": len([t for t in set(topics) if t != -1]),
            "n_outliers_raw": n_outliers_raw,
            "outlier_share_raw": round(n_outliers_raw / len(docs), 4),
            "n_outliers_after_reassignment": n_outliers,
            "outlier_strategy": "reduce_outliers(strategy='embeddings') - nearest topic centroid",
        }

    mix_table = pd.DataFrame(mix_rows_all)
    mix_table.to_csv(
        OUTPUTS_DIR / "tables" / "p4_46_digital_absence_mix__bertopic.csv", index=False
    )
    n_mix = int(mix_table["digital_absence_mix"].sum())
    print(f"\n=== Digital-Absence unforced mix: {n_mix} topic(s) across both corpora ===")

    summary["digital_absence_mixed_topics"] = n_mix
    summary["provenance"] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "script": Path(sys.argv[0]).name,
        "interpreter": sys.executable,
    }
    (OUTPUTS_DIR / "summaries" / "p4_46_bertopic.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    print("=== Done ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
