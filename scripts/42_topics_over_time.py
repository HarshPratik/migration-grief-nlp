"""Phase 5 Task 3 - dynamic topics over time (symbolic crisis).

New.docx asks for BERTopic topics_over_time. The analysis venv bans
umap/hdbscan/bertopic, so the canonical path is the same MiniLM + KMeans
stack already used in scripts/33_phase4_full_compare.py, with topic identity
FIXED globally then tracked across bins (true dynamic topic prevalence).

If BERTopic is importable we also run it. A second comparable method (NMF
fitted independently per calendar year) checks whether Digital and Absence
mix without being forced.

  python scripts/42_topics_over_time.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analysis_config import load_analysis_config, load_anchors  # noqa: E402
from analysis_io import as_token_list, banner, item, out_path, save_summary, save_table  # noqa: E402
from temporal import cfg_temporal  # noqa: E402

TOP_N = 10


def _docs(analysis_dir: Path, label: str) -> pd.DataFrame:
    path = analysis_dir / f"corpus_{label}_temporal.parquet"
    if not path.exists():
        raise SystemExit(f"missing {path}; run scripts/40_temporal_stratify.py first")
    df = pd.read_parquet(path, columns=[
        "post_id", "corpus", "tokens_lemma", "t0_quarter", "cal_quarter",
        "cal_year", "slice_year1", "slice_year3plus", "cal_year1", "cal_year3plus",
        "is_repeat_author",
    ])
    tokens = [as_token_list(t) for t in df["tokens_lemma"]]
    df = df.copy()
    df["tokens"] = tokens
    df["doc"] = [" ".join(t) for t in tokens]
    return df


def _encode(docs: list[str], model_name: str, cache_path: Path) -> np.ndarray:
    if cache_path.exists():
        emb = np.load(cache_path)
        if len(emb) == len(docs):
            item(f"embeddings cache hit {cache_path.name}")
            return emb
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    item(f"encoding {len(docs):,} docs on {device}")
    model = SentenceTransformer(model_name, device=device)
    emb = model.encode(
        docs, batch_size=256, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, emb)
    return emb


def _ctfidf_terms(docs: list[str], labels: np.ndarray, k: int) -> tuple[list[list[str]], list[str]]:
    cluster_docs = []
    for c in range(k):
        members = [docs[i] for i, lab in enumerate(labels) if int(lab) == c]
        cluster_docs.append(" ".join(members) if members else "")
    vec = TfidfVectorizer(max_features=5000, min_df=1, max_df=0.95)
    matrix = vec.fit_transform(cluster_docs)
    vocab = list(vec.get_feature_names_out())
    order = np.argsort(-matrix.toarray(), axis=1)
    terms = []
    for t in range(k):
        terms.append([vocab[i] for i in order[t, :TOP_N]])
    return terms, vocab


def _prevalence(labels: np.ndarray, bins: np.ndarray, k: int) -> pd.DataFrame:
    rows = []
    for b in sorted(set(int(x) for x in bins)):
        mask = bins == b
        n = int(mask.sum())
        if n == 0:
            continue
        counts = np.bincount(labels[mask], minlength=k)
        for t in range(k):
            rows.append({"bin": int(b), "topic": int(t), "n": n, "share": float(counts[t] / n)})
    return pd.DataFrame(rows)


def _domain_mix(terms: list[str], by_domain: dict[str, set[str]]) -> dict[str, int]:
    hits = {d: sum(1 for w in terms if w in vocab) for d, vocab in by_domain.items()}
    hits["digital_absence_mix"] = int(hits.get("digital", 0) > 0 and hits.get("absence", 0) > 0)
    return hits


def _try_bertopic(docs: list[str], timestamps: list, k: int, model_name: str) -> dict | None:
    try:
        from bertopic import BERTopic
        from sentence_transformers import SentenceTransformer
        import torch
    except Exception as exc:
        return {"error": f"bertopic unavailable: {exc}"}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder = SentenceTransformer(model_name, device=device)
    model = BERTopic(
        embedding_model=embedder, nr_topics=k,
        calculate_probabilities=False, verbose=False,
    )
    topics, _ = model.fit_transform(docs, timestamps)
    over_time = model.topics_over_time(docs, timestamps, nr_bins=max(len(set(timestamps)), 2))
    info = model.get_topic_info()
    return {
        "n_topics": int(info.shape[0]),
        "outlier_frac": float((np.asarray(topics) == -1).mean()),
        "over_time": over_time.to_dict(orient="records") if hasattr(over_time, "to_dict") else [],
    }


def _nmf_year(docs: list[str], k: int, seed: int, max_features: int, min_df: int, max_df: float) -> list[list[str]]:
    if len(docs) < max(k, min_df * 2):
        return []
    vec = TfidfVectorizer(max_features=max_features, min_df=min_df, max_df=max_df, lowercase=False)
    matrix = vec.fit_transform(docs)
    if matrix.shape[0] < k or matrix.shape[1] < k:
        return []
    model = NMF(
        n_components=k, init="nndsvd", random_state=seed, max_iter=400,
        solver="cd", beta_loss="frobenius",
    )
    model.fit(matrix)
    vocab = list(vec.get_feature_names_out())
    order = np.argsort(-model.components_, axis=1)
    return [[vocab[i] for i in order[t, :TOP_N]] for t in range(k)]


def main() -> int:
    parser = argparse.ArgumentParser(description="Dynamic topics over time (MiniLM DTM + optional BERTopic)")
    parser.add_argument("--skip-bertopic", action="store_true")
    parser.add_argument("--skip-encode", action="store_true", help="reuse embedding cache only")
    args = parser.parse_args()
    cfg = load_analysis_config()
    anchors = load_anchors()
    knobs = cfg_temporal(cfg)
    k = knobs["topic_k"]
    seed = cfg.seed
    analysis_dir = cfg.path("analysis_dir")
    cache_dir = analysis_dir / "vectors"
    by_domain = {name: {a.term for a in domain.anchors} for name, domain in anchors.domains.items()}
    max_features = int(cfg.get("topics.max_features"))
    min_df = int(cfg.get("topics.min_df"))
    max_df = float(cfg.get("topics.max_df"))

    banner("Dynamic topics")
    item(f"k={k}  MiniLM {knobs['minilm_model']}")
    item("BERTopic umap/hdbscan remain banned in .venv-analysis; MiniLM DTM is the canonical run")

    term_rows = []
    prev_rows = []
    mix_rows = []
    year_rows = []
    bertopic_payload = {}

    for label in ("a", "b"):
        frame = _docs(analysis_dir, label)
        docs = frame["doc"].tolist()
        cache = cache_dir / f"minilm_temporal_{label}.npy"
        if args.skip_encode and not cache.exists():
            raise SystemExit(f"--skip-encode but missing {cache}")
        emb = _encode(docs, knobs["minilm_model"], cache)
        km = MiniBatchKMeans(n_clusters=k, random_state=seed, batch_size=2048, n_init=10)
        labels_arr = km.fit_predict(emb)
        terms, _ = _ctfidf_terms(docs, labels_arr, k)
        for t, words in enumerate(terms):
            hits = _domain_mix(words, by_domain)
            mix_rows.append({"corpus": label, "method": "minilm_global", "topic": t, **hits, "top_terms": " ".join(words)})
            for rank, word in enumerate(words, start=1):
                term_rows.append({
                    "corpus": label, "method": "minilm_global", "topic": t,
                    "rank": rank, "term": word,
                })
        for clock, col in (("t0", "t0_quarter"), ("calendar", "cal_quarter")):
            prev = _prevalence(labels_arr, frame[col].to_numpy(), k)
            prev["corpus"] = label
            prev["clock"] = clock
            prev["method"] = "minilm_global"
            prev_rows.append(prev)

        for year, sub in frame.groupby("cal_year"):
            year_terms = _nmf_year(
                sub["doc"].tolist(), k, seed, max_features, min_df, max_df,
            )
            for t, words in enumerate(year_terms):
                hits = _domain_mix(words, by_domain)
                year_rows.append({
                    "corpus": label, "cal_year": int(year), "n_docs": int(len(sub)),
                    "topic": t, **hits, "top_terms": " ".join(words),
                })

        if not args.skip_bertopic:
            item(f"BERTopic attempt corpus={label}")
            bertopic_payload[label] = _try_bertopic(
                docs, frame["cal_quarter"].tolist(), k, knobs["minilm_model"],
            )

        frame.assign(topic=labels_arr).to_parquet(
            analysis_dir / f"topics_temporal_{label}.parquet", index=False,
        )

    prevalence = pd.concat(prev_rows, ignore_index=True)
    save_table(pd.DataFrame(term_rows), out_path("tables", "p5", 42, "topic_terms"))
    save_table(prevalence, out_path("tables", "p5", 42, "topics_over_time"))
    save_table(pd.DataFrame(mix_rows), out_path("tables", "p5", 42, "domain_mix"))
    save_table(pd.DataFrame(year_rows), out_path("tables", "p5", 42, "nmf_by_calendar_year"))
    mix = pd.DataFrame(mix_rows)
    n_mix = int(mix["digital_absence_mix"].sum()) if not mix.empty else 0
    save_summary(
        {
            "task": "dynamic_topic_modeling",
            "canonical": "minilm_kmeans_global_then_bin_prevalence",
            "bertopic_status": bertopic_payload,
            "k": k,
            "digital_absence_mixed_topics": n_mix,
        },
        out_path("summaries", "p5", 42, "topics_over_time", ext="json"),
    )
    banner("Done")
    item(f"Digital–Absence mixed MiniLM topics: {n_mix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
