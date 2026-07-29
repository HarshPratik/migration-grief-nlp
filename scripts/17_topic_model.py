"""Phase 4 - joint topic model over the POOLED corpus, then corpus contrast.

One model is fitted over all 1,593 posts (906 Corpus A + 687 Corpus B) and the
corpus contrast is read off the document-topic matrix afterwards. Two standalone
models would have to be aligned before they could be compared, and topic
alignment across two models fitted on 906 and 687 short documents is not
defensible at this n - the alignment itself would carry more of the result than
the data does. Pooling makes the topic space identical by construction, so
`mean_theta_a` and `mean_theta_b` are measured on the same ruler.

NMF (default) is deterministic: TF-IDF + `init="nndsvd"` + a fixed
`random_state` has no stochastic component, so two runs produce byte-identical
tables. LDA (counts + `LatentDirichletAllocation`) is offered as the robustness
check, not the primary model.

k is chosen by TERM-CENTRIC STABILITY (Greene, O'Callaghan & Cunningham 2014):
fit the reference model on the full data, refit on `--stability-runs` bootstrap
resamples of the DOCUMENTS, match the refits' topics to the reference with
`linear_sum_assignment` over full topic-term cosine similarity, and take the
mean matched similarity. Salient-term Jaccard is separately required for a
topic to recur; this is a pragmatic Greene-inspired diagnostic rather than an
exact reproduction of every term-set aggregation choice in that paper. Perplexity is deliberately not used: Chang et al. (2009)
showed held-out likelihood anti-correlates with human interpretability, so
optimising it selects models nobody can read.

Two numbers keep the stability figure honest and are always printed beside it:
  * the NULL - mean similarity of the non-matched pairs in the same assignment.
    Matched similarity only means something above it.
  * the CORE-TOPIC FRACTION - the share of reference topics that recur in at
    least `topics.core_topic_threshold` of runs. A high mean agreement with a
    low core fraction means a couple of stable topics are carrying a model whose
    remaining topics are resample noise, and only the core ones are reportable.

Examples:
    D:\\research_paper\\.venv-analysis\\Scripts\\python.exe scripts/17_topic_model.py
    ... scripts/17_topic_model.py --k 8 --no-stability
    ... scripts/17_topic_model.py --method lda --k-sweep 6,8,10
    ... scripts/17_topic_model.py --stability-runs 5 --top-terms 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analysis_config import load_analysis_config, load_anchors  # noqa: E402
from analysis_io import as_token_list, banner, item, make_rng, out_path, save_summary, save_table  # noqa: E402
from vectors import codocument_counts, cosine_matrix, umass_coherence  # noqa: E402

# Smoothing for the corpus-skew log ratio. Only ever binds when a topic has
# essentially zero mass in one corpus, which is reported as `degenerate`.
LOG_RATIO_EPS = 1e-9


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------


def load_pooled(analysis_dir: Path) -> pd.DataFrame:
    """Corpus A and B stacked, in that order. `doc` is the vectorizer input."""
    frames = []
    for label in ("a", "b"):
        path = analysis_dir / f"corpus_{label}.parquet"
        if not path.exists():
            raise SystemExit(f"missing {path}; run scripts/08_reprocess_tokens.py first")
        df = pd.read_parquet(
            path,
            columns=["post_id", "corpus", "subreddit", "username", "word_count", "tokens_lemma"],
        )
        df = df.sort_values("post_id").drop_duplicates("username")
        frame = pd.DataFrame({
            "post_id": df["post_id"].astype(str),
            "corpus": label,
            "subreddit": df["subreddit"].astype(str),
            "username": df["username"].astype(str),
            "word_count": df["word_count"].astype(int),
            "tokens": [as_token_list(t) for t in df["tokens_lemma"]],
        })
        frames.append(frame)
    pooled = pd.concat(frames, ignore_index=True)
    pooled["doc"] = [" ".join(t) for t in pooled["tokens"]]
    return pooled


def vectorize(documents, *, method: str, max_features: int, min_df: int, max_df: float):
    """TF-IDF for NMF, raw counts for LDA - each model's own likelihood assumes it."""
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

    kwargs = dict(max_features=max_features, min_df=min_df, max_df=max_df, lowercase=False)
    vectorizer = TfidfVectorizer(**kwargs) if method == "nmf" else CountVectorizer(**kwargs)
    matrix = vectorizer.fit_transform(documents)
    return matrix, tuple(vectorizer.get_feature_names_out())


# --------------------------------------------------------------------------
# modelling
# --------------------------------------------------------------------------


def fit_topics(matrix, *, method: str, k: int, seed: int, max_iter: int, init: str):
    """Return (doc_topic, topic_term). Deterministic for a fixed seed."""
    if method == "nmf":
        from sklearn.decomposition import NMF

        model = NMF(
            n_components=k, init=init, random_state=seed, max_iter=max_iter,
            solver="cd", beta_loss="frobenius",
        )
        doc_topic = model.fit_transform(matrix)
    else:
        from sklearn.decomposition import LatentDirichletAllocation

        model = LatentDirichletAllocation(
            n_components=k, random_state=seed, max_iter=max_iter,
            learning_method="batch",
        )
        doc_topic = model.fit_transform(matrix)
    return np.asarray(doc_topic, dtype=np.float64), np.asarray(model.components_, dtype=np.float64)


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """Row-normalise a doc-topic matrix to proportions. All-zero rows stay zero."""
    totals = matrix.sum(axis=1, keepdims=True)
    return matrix / np.where(totals > 0, totals, 1.0)


def top_term_indices(topic_term: np.ndarray, n: int) -> np.ndarray:
    """Indices of the n highest-weighted terms per topic, highest first."""
    order = np.argsort(-topic_term, axis=1, kind="stable")
    return order[:, :n]


def top_terms(topic_term: np.ndarray, vocabulary, n: int) -> list[list[str]]:
    return [[vocabulary[i] for i in row] for row in top_term_indices(topic_term, n)]


# --------------------------------------------------------------------------
# stability (Greene, O'Callaghan & Cunningham 2014)
# --------------------------------------------------------------------------


def stability_for_k(
    matrix,
    *,
    k: int,
    runs: int,
    method: str,
    seed: int,
    max_iter: int,
    init: str,
    vocabulary,
    n_top: int,
) -> dict:
    """Bootstrap the documents, refit, match topics to the reference model.

    Agreement is the mean cosine of the matched topic-term pairs. Reported with
    the mean cosine of the same assignment's NON-matched pairs, which is the
    floor any k-topic model reaches by chance on this term space.
    """
    from scipy.optimize import linear_sum_assignment

    n_docs = matrix.shape[0]
    _, reference = fit_topics(matrix, method=method, k=k, seed=seed, max_iter=max_iter, init=init)
    reference_terms = [set(t) for t in top_terms(reference, vocabulary, n_top)]

    rng = make_rng(seed + k)
    matched_per_run: list[np.ndarray] = []
    jaccard_per_run: list[np.ndarray] = []
    null_scores: list[float] = []

    for _ in range(runs):
        index = rng.integers(0, n_docs, size=n_docs)
        _, sampled = fit_topics(
            matrix[index], method=method, k=k, seed=seed, max_iter=max_iter, init=init
        )
        similarity = cosine_matrix(reference, sampled)
        rows, cols = linear_sum_assignment(-similarity)
        run_matched = np.zeros(k, dtype=np.float64)
        run_matched[rows] = similarity[rows, cols]
        matched_per_run.append(run_matched)

        off = similarity.copy()
        off[rows, cols] = np.nan
        null_scores.append(float(np.nanmean(off)))

        sampled_terms = [set(t) for t in top_terms(sampled, vocabulary, n_top)]
        # Assignment returns rows in arbitrary order in principle; store both
        # diagnostics by reference-topic index so recurrence columns align.
        run_jaccard = np.zeros(k, dtype=np.float64)
        run_jaccard[rows] = np.array([
            len(reference_terms[r] & sampled_terms[c]) / max(len(reference_terms[r] | sampled_terms[c]), 1)
            for r, c in zip(rows, cols)
        ])
        jaccard_per_run.append(run_jaccard)

    matched = np.vstack(matched_per_run)        # runs x k
    jaccard = np.vstack(jaccard_per_run)
    # A topic recurs only when (a) its matched cosine beats that run's measured
    # non-matched null and (b) at least half of its salient top terms repeat.
    # The term-set condition prevents a generic long-tailed weight vector from
    # being certified as core merely because assignment found its best match.
    null_array = np.asarray(null_scores, dtype=np.float64)
    recurrence = ((matched > null_array[:, None]) & (jaccard >= 0.5)).mean(axis=0)

    return {
        "k": k,
        "n_runs": runs,
        "mean_agreement": float(matched.mean()),
        "sd_agreement": float(matched.mean(axis=1).std()),
        "mean_topterm_jaccard": float(jaccard.mean()),
        "null_agreement": float(np.mean(null_scores)),
        "recurrence": recurrence,
        "per_topic_agreement": matched.mean(axis=0),
    }


# --------------------------------------------------------------------------
# corpus contrast
# --------------------------------------------------------------------------


def prevalence_table(
    theta: np.ndarray,
    corpus: np.ndarray,
    word_count: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    """Mean theta per corpus, log ratio, percentile bootstrap CI.

    Documents are the resampling unit (config `stats.resampling_unit: post`);
    resampling topics or tokens would treat words inside one post as independent
    observations, which they are not.
    """
    is_a = corpus == "a"
    theta_a, theta_b = theta[is_a], theta[~is_a]
    mean_a, mean_b = theta_a.mean(axis=0), theta_b.mean(axis=0)

    rng = make_rng(seed)
    strata = pd.qcut(word_count, 10, labels=False, duplicates="drop")
    strata_a, strata_b = strata[is_a], strata[~is_a]
    draws = np.empty((n_bootstrap, theta.shape[1]), dtype=np.float64)
    for draw in range(n_bootstrap):
        sampled = []
        for values, groups in ((theta_a, strata_a), (theta_b, strata_b)):
            indices = []
            for stratum in np.unique(groups):
                members = np.flatnonzero(groups == stratum)
                indices.extend(rng.choice(members, size=len(members), replace=True))
            sampled.append(values[np.asarray(indices)].mean(axis=0))
        draws[draw] = np.log((sampled[0] + LOG_RATIO_EPS) / (sampled[1] + LOG_RATIO_EPS))

    low, high = np.percentile(draws, [2.5, 97.5], axis=0)
    log_ratio = np.log((mean_a + LOG_RATIO_EPS) / (mean_b + LOG_RATIO_EPS))

    return pd.DataFrame({
        "topic": np.arange(theta.shape[1]),
        "mean_theta_a": mean_a,
        "mean_theta_b": mean_b,
        "log_ratio": log_ratio,
        "ci_low": low,
        "ci_high": high,
        "ci_excludes_zero": (low > 0) | (high < 0),
        "degenerate": (mean_a < LOG_RATIO_EPS) | (mean_b < LOG_RATIO_EPS),
    })


def anchor_counts(terms: list[str], domain_of: dict[str, str], domains) -> dict[str, int]:
    counts = {name: 0 for name in domains}
    for term in terms:
        domain = domain_of.get(term)
        if domain is not None:
            counts[domain] += 1
    return counts


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def main() -> int:
    cfg = load_analysis_config()
    defaults = cfg.section("topics")

    parser = argparse.ArgumentParser(description="Pooled topic model with corpus-conditional prevalence")
    parser.add_argument("--method", choices=["nmf", "lda"], default=str(defaults["method"]))
    parser.add_argument("--k", type=int, default=None,
                        help="number of topics; default is the stability peak, "
                             f"or topics.k ({defaults['k']}) when stability is off")
    parser.add_argument("--k-sweep", default=",".join(str(x) for x in defaults["k_sweep"]),
                        help="comma-separated k values to test for stability")
    parser.add_argument("--stability-runs", type=int, default=int(defaults["stability_runs"]))
    parser.add_argument("--top-terms", type=int, default=int(defaults["top_terms"]))
    parser.add_argument("--max-features", type=int, default=int(defaults["max_features"]))
    parser.add_argument("--min-df", type=int, default=int(defaults["min_df"]))
    parser.add_argument("--max-df", type=float, default=float(defaults["max_df"]))
    parser.add_argument("--analysis-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-stability", action="store_true",
                        help="skip the k sweep (fast path; k selection is then unjustified)")
    args = parser.parse_args()

    if args.k is not None and args.k < 2:
        parser.error("--k must be >= 2")
    if args.stability_runs < 1:
        parser.error("--stability-runs must be >= 1")
    if args.top_terms < 2:
        parser.error("--top-terms must be >= 2")
    if args.max_features < 2:
        parser.error("--max-features must be >= 2")
    if args.min_df < 1:
        parser.error("--min-df must be >= 1")
    if not 0.0 < args.max_df <= 1.0:
        parser.error("--max-df must be in (0, 1]")

    cfg = load_analysis_config(seed=args.seed)
    anchors = load_anchors()
    seed = cfg.seed
    analysis_dir = args.analysis_dir or cfg.path("analysis_dir")
    max_iter = int(cfg.get("topics.max_iter", 500))
    init = str(cfg.get("topics.nmf_init", "nndsvd"))
    core_threshold = float(cfg.get("topics.core_topic_threshold", 0.8))
    n_bootstrap = int(cfg.get("stats.n_bootstrap", 2000))

    try:
        sweep = [int(x) for x in str(args.k_sweep).split(",") if str(x).strip()]
    except ValueError:
        parser.error(f"--k-sweep: expected comma-separated integers, got {args.k_sweep!r}")
    if not sweep or any(k < 2 for k in sweep):
        parser.error("--k-sweep must contain only integers >= 2")

    pooled = load_pooled(analysis_dir)
    matrix, vocabulary = vectorize(
        pooled["doc"], method=args.method,
        max_features=args.max_features, min_df=args.min_df, max_df=args.max_df,
    )
    corpus = pooled["corpus"].to_numpy()
    max_k = min(matrix.shape)
    if args.k is not None and args.k > max_k:
        parser.error(f"--k must be <= min(matrix shape) = {max_k}")

    banner("Pooled corpus")
    item(f"documents    {len(pooled)} ({int((corpus == 'a').sum())} A + {int((corpus == 'b').sum())} B), modelled JOINTLY")
    item(f"matrix       {matrix.shape[0]} x {matrix.shape[1]} ({args.method.upper()}: "
         f"{'tf-idf' if args.method == 'nmf' else 'counts'}), density {matrix.nnz / (matrix.shape[0] * matrix.shape[1]):.4f}")
    item(f"vocabulary   min_df={args.min_df}, max_df={args.max_df}, max_features={args.max_features}")
    item(f"seed         {seed}, init={init if args.method == 'nmf' else 'n/a'}, max_iter={max_iter}")
    empty_docs = int((np.asarray(matrix.sum(axis=1)).ravel() == 0).sum())
    if empty_docs:
        item(f"WARNING: {empty_docs} document(s) have no in-vocabulary term; their theta is all-zero.")

    # ---- k selection ------------------------------------------------------
    sweep_rows: list[dict] = []
    stability: dict[int, dict] = {}
    chosen_from = "config"

    if not args.no_stability:
        banner(f"Term-centric stability sweep ({args.stability_runs} bootstrap refits per k)")
        item(f"{'k':>4}{'agreement':>12}{'sd':>8}{'null':>8}{'jaccard':>9}{'core':>8}")
        for k in sweep:
            if k > max_k:
                item(f"{k:>4}  skipped - k exceeds min(matrix shape) ({max_k})")
                continue
            result = stability_for_k(
                matrix, k=k, runs=args.stability_runs, method=args.method, seed=seed,
                max_iter=max_iter, init=init, vocabulary=vocabulary, n_top=args.top_terms,
            )
            stability[k] = result
            core_fraction = float((result["recurrence"] >= core_threshold).mean())
            sweep_rows.append({
                "k": k,
                "n_runs": result["n_runs"],
                "mean_agreement": result["mean_agreement"],
                "sd_agreement": result["sd_agreement"],
                "null_agreement": result["null_agreement"],
                "agreement_above_null": result["mean_agreement"] - result["null_agreement"],
                "mean_topterm_jaccard": result["mean_topterm_jaccard"],
                "core_topic_fraction": core_fraction,
                "n_core_topics": int((result["recurrence"] >= core_threshold).sum()),
            })
            item(f"{k:>4}{result['mean_agreement']:>12.3f}{result['sd_agreement']:>8.3f}"
                 f"{result['null_agreement']:>8.3f}{result['mean_topterm_jaccard']:>9.3f}"
                 f"{core_fraction:>8.2f}")

        if not sweep_rows:
            raise SystemExit("--k-sweep produced no usable k values")
        peak = max(sweep_rows, key=lambda r: (r["agreement_above_null"], -r["k"]))
        item(f"baseline-adjusted stability peak at k={peak['k']} "
             f"(agreement {peak['mean_agreement']:.3f}, null {peak['null_agreement']:.3f}, "
             f"delta {peak['agreement_above_null']:.3f})")
        chosen_from = "baseline-adjusted stability peak"
    else:
        peak = None

    if args.k is not None:
        k = args.k
        chosen_from = "--k"
    elif peak is not None:
        k = int(peak["k"])
    else:
        k = int(defaults["k"])

    banner(f"Final model: {args.method.upper()}, k={k} ({chosen_from})")
    if args.method == "nmf":
        item("theta values are unit-sum normalized TF-IDF reconstruction loadings, not "
             "identified topic probabilities")

    if k not in stability and not args.no_stability:
        stability[k] = stability_for_k(
            matrix, k=k, runs=args.stability_runs, method=args.method, seed=seed,
            max_iter=max_iter, init=init, vocabulary=vocabulary, n_top=args.top_terms,
        )

    doc_topic, topic_term = fit_topics(
        matrix, method=args.method, k=k, seed=seed, max_iter=max_iter, init=init
    )
    theta = normalize_rows(doc_topic)
    empty_mask = np.asarray(matrix.sum(axis=1)).ravel() == 0
    theta[empty_mask] = 0.0
    terms_per_topic = top_terms(topic_term, vocabulary, args.top_terms)

    row_sums = theta.sum(axis=1)
    non_empty = row_sums > 0
    item(f"theta rows sum to 1: {int(np.isclose(row_sums[non_empty], 1.0).sum())}/{int(non_empty.sum())} "
         f"non-empty documents (max deviation {float(np.abs(row_sums[non_empty] - 1).max()):.2e})")

    # ---- coherence, anchors ------------------------------------------------
    documents = list(pooled["tokens"])
    wanted = sorted({t for terms in terms_per_topic for t in terms})
    df_counts, codf_counts = codocument_counts(documents, wanted)
    coherence = [umass_coherence(terms, df_counts, codf_counts) for terms in terms_per_topic]

    domain_of = {a.term: a.domain for a in anchors.anchors}
    domain_names = list(anchors.domains)
    per_topic_anchors = [anchor_counts(terms, domain_of, domain_names) for terms in terms_per_topic]

    # ---- corpus contrast ---------------------------------------------------
    prevalence = prevalence_table(
        theta,
        corpus,
        pooled["word_count"].to_numpy(),
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    prevalence["top_terms"] = [" ".join(t) for t in terms_per_topic]
    prevalence["coherence_umass"] = coherence
    prevalence["n_anchors_total"] = [sum(c.values()) for c in per_topic_anchors]
    prevalence["n_anchors_by_domain"] = [
        ";".join(f"{name}={counts[name]}" for name in domain_names if counts[name])
        for counts in per_topic_anchors
    ]
    for name in domain_names:
        prevalence[f"n_anchors_{name}"] = [counts[name] for counts in per_topic_anchors]

    if k in stability:
        prevalence["recurrence"] = stability[k]["recurrence"]
        prevalence["is_core"] = stability[k]["recurrence"] >= core_threshold
    else:
        prevalence["recurrence"] = np.nan
        prevalence["is_core"] = False

    # ---- report ------------------------------------------------------------
    banner("Topics (top 8 terms), pooled model")
    for _, row in prevalence.iterrows():
        marker = "core" if bool(row["is_core"]) else "    "
        item(f"T{int(row['topic']):<3}{marker} {' '.join(row['top_terms'].split()[:8])}")
        item(f"     A={row['mean_theta_a']:.4f}  B={row['mean_theta_b']:.4f}  "
             f"log-ratio={row['log_ratio']:+.3f} [{row['ci_low']:+.3f}, {row['ci_high']:+.3f}]"
             f"{'  *' if bool(row['ci_excludes_zero']) else ''}"
             f"  umass={row['coherence_umass']:.2f}"
             f"{'  anchors: ' + row['n_anchors_by_domain'] if row['n_anchors_by_domain'] else ''}",
             indent=2)

    banner("Core topics")
    if k in stability:
        recurrence = stability[k]["recurrence"]
        core_fraction = float((recurrence >= core_threshold).mean())
        item("a topic recurs when matched cosine beats the run's non-matched null AND "
             "top-term Jaccard >= 0.5")
        item(f"a topic is CORE when it recurs in >= {core_threshold:.0%} of "
             f"{args.stability_runs} bootstrap refits")
        item(f"core topics: {int((recurrence >= core_threshold).sum())}/{k} "
             f"= {core_fraction:.0%} of the model")
        item(f"mean agreement {stability[k]['mean_agreement']:.3f} vs null "
             f"{stability[k]['null_agreement']:.3f}")
        if core_fraction < 0.5:
            item("HEADLINE: most topics in this model do NOT survive document resampling.")
            item("Only the topics marked `core` above are reportable as findings; the rest are "
                 "an artefact of this particular sample of posts and must not be interpreted.", indent=2)
        elif core_fraction < 0.8:
            item("CAUTION: a substantial minority of topics do not survive resampling. "
                 "Interpret only the `core` ones.")
    else:
        core_fraction = float("nan")
        item("stability not run (--no-stability); k is unjustified and no topic is certified core")

    banner("Corpus skew (positive log-ratio = larger loading share in Corpus A)")
    significant = prevalence[prevalence["ci_excludes_zero"] & prevalence["is_core"]]
    item("intervals are exploratory and conditional on this fitted, selected topic solution; "
         "they do not include model or k-selection uncertainty")
    item(f"{len(significant)} core topic(s) have a nominal 95% conditional bootstrap CI excluding zero "
         "(non-core topics are not presented as findings)")
    for _, row in significant.sort_values("log_ratio", ascending=False).iterrows():
        side = "A" if row["log_ratio"] > 0 else "B"
        item(f"T{int(row['topic']):<3} -> {side}  {row['log_ratio']:+.3f} "
             f"[{row['ci_low']:+.3f}, {row['ci_high']:+.3f}]  {' '.join(row['top_terms'].split()[:6])}")
    if bool(prevalence["degenerate"].any()):
        item("WARNING: at least one topic has ~zero mass in a corpus; its log-ratio is "
             "an artefact of the smoothing constant, not a measurement.")

    # ---- outputs -----------------------------------------------------------
    written: list[str] = []
    topics_dir = analysis_dir / "topics"
    topics_dir.mkdir(parents=True, exist_ok=True)
    doctopic_path = topics_dir / f"{args.method}_k{k}_doctopic.npy"
    np.save(doctopic_path, theta)
    written.append(str(doctopic_path))

    term_index = {term: i for i, term in enumerate(vocabulary)}
    terms_long = pd.DataFrame([
        {"topic": t, "rank": r + 1, "term": term,
         "weight": float(topic_term[t, term_index[term]]),
         "is_anchor": term in domain_of, "domain": domain_of.get(term, "")}
        for t, terms in enumerate(terms_per_topic)
        for r, term in enumerate(terms)
    ])
    written += [str(p) for p in save_table(
        terms_long, out_path("tables", "p4", 17, "topic_terms", f"{args.method}_k{k}"), parquet=False
    )]

    doc_table = pd.DataFrame({
        "post_id": pooled["post_id"],
        "corpus": pooled["corpus"],
        "subreddit": pooled["subreddit"],
        "dominant_topic": np.where(row_sums > 0, theta.argmax(axis=1), -1),
        "dominant_theta": theta.max(axis=1),
        "theta_sum": row_sums,
    })
    for t in range(k):
        doc_table[f"theta_{t:02d}"] = theta[:, t]
    written += [str(p) for p in save_table(
        doc_table, out_path("tables", "p4", 17, "doc_topics", f"{args.method}_k{k}"), parquet=False
    )]

    written += [str(p) for p in save_table(
        prevalence, out_path("tables", "p4", 17, "topic_prevalence", f"{args.method}_k{k}"), parquet=False
    )]

    if sweep_rows:
        written += [str(p) for p in save_table(
            pd.DataFrame(sweep_rows), out_path("tables", "p4", 17, "k_sweep", args.method), parquet=False
        )]

    summary = {
        "method": args.method,
        "scope": "pooled",
        "k": k,
        "k_selected_by": chosen_from,
        "k_sweep": sweep_rows,
        "stability_runs": args.stability_runs if not args.no_stability else 0,
        "recurrence_definition": "matched full-vector cosine > run mean non-matched cosine AND top-term Jaccard >= 0.5",
        "core_topic_threshold": core_threshold,
        "contrast_interval_scope": "nominal 95% word-count-decile-stratified post bootstrap after one-post-per-author capping; conditional on fitted selected topic solution",
        "author_cluster_policy": "cap_one_per_author",
        "nmf_theta_interpretation": "unit-sum normalized TF-IDF reconstruction loadings; descriptive shares, not identified probabilities",
        "core_topic_fraction": None if np.isnan(core_fraction) else round(core_fraction, 4),
        "mean_agreement": stability[k]["mean_agreement"] if k in stability else None,
        "null_agreement": stability[k]["null_agreement"] if k in stability else None,
        "n_documents": int(len(pooled)),
        "n_documents_a": int((corpus == "a").sum()),
        "n_documents_b": int((corpus == "b").sum()),
        "n_empty_documents": empty_docs,
        "vocabulary_size": len(vocabulary),
        "vectorizer": {
            "kind": "tfidf" if args.method == "nmf" else "count",
            "max_features": args.max_features, "min_df": args.min_df, "max_df": args.max_df,
        },
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "topics": [
            {
                "topic": int(row["topic"]),
                "top_terms": row["top_terms"].split(),
                "mean_theta_a": float(row["mean_theta_a"]),
                "mean_theta_b": float(row["mean_theta_b"]),
                "log_ratio": float(row["log_ratio"]),
                "ci": [float(row["ci_low"]), float(row["ci_high"])],
                "ci_excludes_zero": bool(row["ci_excludes_zero"]),
                "coherence_umass": float(row["coherence_umass"]),
                "recurrence": None if pd.isna(row["recurrence"]) else float(row["recurrence"]),
                "is_core": bool(row["is_core"]),
                "anchors_by_domain": {
                    name: int(row[f"n_anchors_{name}"]) for name in domain_names
                },
            }
            for _, row in prevalence.iterrows()
        ],
        "outputs": written,
    }
    save_summary(summary, out_path("summaries", "p4", 17, "topic_model", ext="json"))

    banner("Done")
    item(f"doc-topic -> {doctopic_path.relative_to(ROOT)}  shape {theta.shape}")
    item(f"summary   -> {out_path('summaries', 'p4', 17, 'topic_model', ext='json').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
