"""Phase 3 - aspect-based emotion classification over anchor context windows.

Implements phases/Phase 3.docx, which asks for a pre-trained EMOTION classifier
("instead of basic Positive/Negative sentiment"), aspect-based analysis, and
statistical comparison of the emotional distributions between corpora.

Primary model: j-hartmann/emotion-english-distilroberta-base - 7 classes
(anger, disgust, fear, joy, neutral, sadness, surprise), a superset of the five
the brief names, and a DistilBERT-family model as suggested. Full probability
vectors are stored, not just the argmax, so intensity analyses are possible
later without re-running the model.

VADER is retained but DEMOTED to a convergent check: it emits valence only and
cannot answer the question asked. It is useful as a wiring test - if the
classifier's sadness/fear mass does not move opposite to VADER compound,
something is misconnected.

ABSA CAVEAT, stated plainly because it bounds every claim made from this
output: this is a WINDOW PROXY for aspect-based analysis. True ABSA models need
aspect-annotated training data that does not exist for bereavement or migration
narrative, so emotion is classified over the anchor's local context and the
anchor is treated as the aspect. A window can carry an emotion the aspect does
not own ("I was furious that I missed her phone call").

Three corrections to the statistics the brief requests:
  1. CHI-SQUARE IS INVALID at these cell counts. A 2x7 emotion-by-corpus table
     for a Tier-2 anchor has several near-zero expected cells. Where any
     expected count is below `stats.min_expected_cell`, this script runs a
     Monte-Carlo permuted chi-square instead and records which test was used.
  2. INDEPENDENCE IS VIOLATED - one post contributes many windows and posts
     contribute unequally. Emotion proportions are aggregated per (post, anchor)
     FIRST, then compared at post level. This also neutralises Corpus B's
     greater post length.
  3. A T-TEST IS THE WRONG SHAPE for a 7-class distribution. Per-emotion
     differences get bootstrap CIs, and the distributions as a whole are
     compared with Jensen-Shannon distance under a permutation null.

Example:
    ...\\python.exe scripts/15_emotion_absa.py --limit 500        # quick check
    ...\\python.exe scripts/15_emotion_absa.py                    # full run
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
from analysis_io import banner, item, make_rng, out_path, save_summary, save_table  # noqa: E402

CORPORA = ("a", "b")


# --------------------------------------------------------------------------
# classification
# --------------------------------------------------------------------------


def classify_emotions(texts: list[str], model_name: str, batch_size: int) -> tuple[np.ndarray, list[str]]:
    """Full probability matrix (n_texts, n_labels) plus the label order."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    labels = [model.config.id2label[i].lower() for i in range(model.config.num_labels)]

    out = np.zeros((len(texts), len(labels)), dtype=np.float32)
    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        encoded = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            logits = model(**encoded).logits
        out[start : start + len(chunk)] = torch.softmax(logits, dim=-1).numpy()
    return out, labels


def vader_compound(texts: list[str]) -> np.ndarray:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    analyzer = SentimentIntensityAnalyzer()
    return np.array([analyzer.polarity_scores(t)["compound"] for t in texts], dtype=np.float32)


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------


def jensen_shannon(p: np.ndarray, q: np.ndarray) -> float:
    """JS distance (sqrt of JS divergence), base 2, bounded [0, 1]."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p / max(p.sum(), 1e-12)
    q = q / max(q.sum(), 1e-12)
    m = 0.5 * (p + q)

    def _kl(x, y):
        mask = x > 0
        return float(np.sum(x[mask] * np.log2(x[mask] / np.maximum(y[mask], 1e-12))))

    return float(np.sqrt(max(0.5 * _kl(p, m) + 0.5 * _kl(q, m), 0.0)))


def contingency_test(table: np.ndarray, *, min_expected: int, rng, n_permutations: int) -> dict:
    """Chi-square where valid, fixed-margin Monte Carlo chi-square otherwise."""
    from scipy.stats import chi2_contingency

    counts = np.asarray(table, dtype=int)
    counts = counts[:, counts.sum(axis=0) > 0]
    if counts.shape[1] < 2 or counts.sum() == 0:
        return {"test": "none", "statistic": np.nan, "p_value": np.nan, "min_expected": 0.0}

    stat, p, _, expected = chi2_contingency(counts, correction=False)
    smallest = float(expected.min())
    if smallest >= min_expected:
        return {"test": "chi2", "statistic": float(stat), "p_value": float(p), "min_expected": smallest}

    # Conditional null with both margins fixed. For a 2xK table, drawing the
    # first row from a multivariate hypergeometric determines the second row.
    column_totals = counts.sum(axis=0)
    first_rows = rng.multivariate_hypergeometric(
        column_totals, int(counts[0].sum()), size=n_permutations
    )
    simulated = np.stack([first_rows, column_totals - first_rows], axis=1)
    simulated_stats = np.sum((simulated - expected) ** 2 / expected, axis=(1, 2))
    return {
        "test": "monte_carlo_chi2",
        "statistic": float(stat),
        "p_value": float((np.count_nonzero(simulated_stats >= stat) + 1) / (n_permutations + 1)),
        "min_expected": smallest,
    }


def stratified_bootstrap_means(
    values: np.ndarray,
    word_count: np.ndarray,
    *,
    n_bootstrap: int,
    rng,
) -> np.ndarray:
    """Bootstrap rows within word-count deciles and return mean vectors."""
    strata = pd.qcut(word_count, 10, labels=False, duplicates="drop")
    draws = np.empty((n_bootstrap, values.shape[1]), dtype=float)
    for draw in range(n_bootstrap):
        indices = []
        for stratum in np.unique(strata):
            members = np.flatnonzero(strata == stratum)
            indices.extend(rng.choice(members, size=len(members), replace=True))
        draws[draw] = values[np.asarray(indices)].mean(axis=0)
    return draws


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    valid = ~np.isnan(p)
    out = np.full(len(p), np.nan)
    if not valid.any():
        return out
    sub = p[valid]
    order = np.argsort(sub)
    ranked = sub[order]
    q = ranked * len(sub) / (np.arange(len(sub)) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    restored = np.empty_like(q)
    restored[order] = np.clip(q, 0, 1)
    out[valid] = restored
    return out


# --------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Emotion classification and ABSA over anchor windows")
    parser.add_argument("--corpus", choices=["a", "b", "both"], default="both")
    parser.add_argument("--model", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--window-size", type=int, default=None)
    parser.add_argument("--limit", type=int, default=0, help="classify only the first N windows (debug)")
    parser.add_argument("--no-vader", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = load_analysis_config(seed=args.seed)
    anchors = load_anchors()
    analysis_dir = cfg.path("analysis_dir")
    model_name = args.model or str(cfg.get("emotion.model"))
    batch_size = args.batch_size or int(cfg.get("emotion.batch_size"))
    window_size = args.window_size or int(cfg.get("windows.size"))
    min_expected = int(cfg.get("stats.min_expected_cell"))
    n_permutations = int(cfg.get("stats.n_permutations"))
    n_boot = int(cfg.get("stats.n_bootstrap"))
    author_labels = [str(x) for x in cfg.get("emotion.author_labels")]
    rng = make_rng(cfg.seed)
    tier_path = out_path("tables", "p2", 9, "anchor_coverage")
    if not tier_path.exists():
        raise SystemExit(f"missing {tier_path}; run scripts/09_anchor_coverage.py first")
    tier_table = pd.read_csv(tier_path)
    anchor_tier = dict(zip(tier_table["anchor"].astype(str), tier_table["tier_observed"].astype(int)))
    polysemous = {anchor.term for anchor in anchors.anchors if anchor.polysemous}

    labels_to_run = ["a", "b"] if args.corpus == "both" else [args.corpus]
    frames = []
    for label in labels_to_run:
        path = analysis_dir / f"windows_{label}.parquet"
        if not path.exists():
            raise SystemExit(f"missing {path}; run scripts/11_extract_windows.py first")
        frame = pd.read_parquet(path)
        if "window_size" in frame.columns:
            frame = frame[frame["window_size"] == window_size]
        frames.append(frame)
    windows = pd.concat(frames, ignore_index=True)
    if args.limit:
        windows = windows.head(args.limit)

    banner("Emotion classification")
    item(f"model         {model_name}")
    item(f"windows       {len(windows)} at +/-{window_size} tokens")
    item(f"corpora       {sorted(windows['corpus'].unique())}")

    texts = windows["window_text"].fillna("").astype(str).tolist()
    probabilities, emotion_labels = classify_emotions(texts, model_name, batch_size)
    item(f"labels        {emotion_labels}")

    scored = windows.copy()
    for i, name in enumerate(emotion_labels):
        scored[f"p_{name}"] = probabilities[:, i]
    scored["emotion"] = [emotion_labels[i] for i in probabilities.argmax(axis=1)]

    if not args.no_vader and bool(cfg.get("emotion.vader_check")):
        scored["vader_compound"] = vader_compound(texts)
        negative_mass = scored[[c for c in (f"p_sadness", f"p_fear") if c in scored]].sum(axis=1)
        correlation = float(np.corrcoef(negative_mass, scored["vader_compound"])[0, 1])
        item(f"VADER check   corr(sadness+fear, compound) = {correlation:+.3f} (expect negative)")
        if correlation > 0:
            item("WARNING: expected a negative correlation; check the pipeline wiring", indent=2)

    if args.limit:
        item("debug limit active: classification validated; no canonical outputs written")
        return 0

    for label in labels_to_run:
        subset = scored[scored["corpus"] == label]
        save_table(
            subset,
            analysis_dir / f"emotion_{label}_{model_name.split('/')[-1]}",
            parquet=True, csv=False,
        )

    # ---- aggregate to post level (independence fix) -----------------------
    probability_columns = [f"p_{name}" for name in emotion_labels]
    metadata = []
    for label in labels_to_run:
        frame = pd.read_parquet(
            analysis_dir / f"corpus_{label}.parquet",
            columns=["post_id", "username", "word_count"],
        )
        metadata.append(frame.assign(corpus=label))
    post_metadata = pd.concat(metadata, ignore_index=True)
    post_metadata = post_metadata.sort_values("post_id").drop_duplicates(["corpus", "username"])
    allowed_posts = set(post_metadata["post_id"].astype(str))
    scored_for_inference = scored[
        scored["post_id"].astype(str).isin(allowed_posts) & ~scored["anchor"].isin(polysemous)
    ].copy()

    # Domain rows must collapse across anchors as well as repeated occurrences:
    # one post contributes one vector to a domain comparison, never one per
    # matching anchor. Anchor rows retain the anchor in their grouping key.
    per_post = {
        "domain": (
            scored_for_inference.groupby(["corpus", "domain", "post_id"])[probability_columns]
            .mean()
            .reset_index()
            .merge(post_metadata[["corpus", "post_id", "word_count"]], on=["corpus", "post_id"])
        ),
        "anchor": (
            scored_for_inference.groupby(["corpus", "anchor", "post_id"])[probability_columns]
            .mean()
            .reset_index()
            .merge(post_metadata[["corpus", "post_id", "word_count"]], on=["corpus", "post_id"])
        ),
    }

    # ---- per-domain and per-anchor comparison ----------------------------
    results = []
    for level, key in (("domain", "domain"), ("anchor", "anchor")):
        for name, group in per_post[level].groupby(key):
            tier = 1 if level == "domain" else anchor_tier.get(str(name), 3)
            if tier == 3:
                continue
            a = group[group["corpus"] == "a"]
            b = group[group["corpus"] == "b"]
            if len(a) < 3 or len(b) < 3:
                continue
            inferential = tier == 1
            vectors_a = a[probability_columns].to_numpy()
            vectors_b = b[probability_columns].to_numpy()
            mean_a = vectors_a.mean(axis=0)
            mean_b = vectors_b.mean(axis=0)

            # A hard-label table is retained as an interpretable diagnostic,
            # derived after post aggregation so every post contributes once.
            counts = np.vstack([
                np.bincount(vectors_a.argmax(axis=1), minlength=len(emotion_labels)),
                np.bincount(vectors_b.argmax(axis=1), minlength=len(emotion_labels)),
            ])
            test = (
                contingency_test(
                    counts,
                    min_expected=min_expected,
                    rng=rng,
                    n_permutations=n_permutations,
                )
                if inferential
                else {"test": "descriptive_only", "statistic": np.nan,
                      "p_value": np.nan, "min_expected": np.nan}
            )

            observed_js = jensen_shannon(mean_a, mean_b)
            pooled = np.vstack([vectors_a, vectors_b])
            lengths = np.concatenate([a["word_count"].to_numpy(), b["word_count"].to_numpy()])
            strata = pd.qcut(lengths, 10, labels=False, duplicates="drop")
            null = []
            if inferential:
                observed_groups = np.concatenate([np.zeros(len(a), dtype=int), np.ones(len(b), dtype=int)])
                for _ in range(n_permutations):
                    permuted_groups = observed_groups.copy()
                    for stratum in np.unique(strata):
                        indices = np.flatnonzero(strata == stratum)
                        permuted_groups[indices] = rng.permutation(permuted_groups[indices])
                    null.append(jensen_shannon(
                        pooled[permuted_groups == 0].mean(0),
                        pooled[permuted_groups == 1].mean(0),
                    ))
            js_p = (
                float((np.sum(np.asarray(null) >= observed_js) + 1) / (len(null) + 1))
                if null else np.nan
            )

            bootstrap_a = stratified_bootstrap_means(
                vectors_a, a["word_count"].to_numpy(), n_bootstrap=n_boot, rng=rng
            )
            bootstrap_b = stratified_bootstrap_means(
                vectors_b, b["word_count"].to_numpy(), n_bootstrap=n_boot, rng=rng
            )
            record = {
                "level": level, "target": name, "tier": tier,
                "in_fdr_family": inferential,
                "n_posts_a": len(a), "n_posts_b": len(b),
                "n_windows_a": int(
                    len(scored_for_inference[(scored_for_inference[key] == name) & (scored_for_inference["corpus"] == "a")])
                ),
                "n_windows_b": int(
                    len(scored_for_inference[(scored_for_inference[key] == name) & (scored_for_inference["corpus"] == "b")])
                ),
                "js_distance": round(observed_js, 4), "js_p": round(js_p, 5),
                "test": test["test"], "statistic": round(test["statistic"], 3),
                "p_value": round(test["p_value"], 5), "min_expected_cell": round(test["min_expected"], 2),
            }
            for i, emotion in enumerate(emotion_labels):
                record[f"a_{emotion}"] = round(float(mean_a[i]), 4)
                record[f"b_{emotion}"] = round(float(mean_b[i]), 4)
                draws = bootstrap_a[:, i] - bootstrap_b[:, i]
                low, high = np.percentile(draws, [2.5, 97.5])
                record[f"diff_{emotion}"] = round(float(mean_a[i] - mean_b[i]), 4)
                record[f"ci_{emotion}"] = f"[{low:+.3f}, {high:+.3f}]"
            results.append(record)

    table = pd.DataFrame(results)
    if not table.empty:
        table["fdr_q"] = np.nan
        family = table["in_fdr_family"]
        table.loc[family, "fdr_q"] = benjamini_hochberg(table.loc[family, "js_p"].to_numpy())
        table = table.sort_values(["level", "js_distance"], ascending=[True, False])
        save_table(table, out_path("tables", "p3", 15, "emotion_comparison"), parquet=False)

        matrix_columns = ["level", "target", "n_windows_a", "n_windows_b"] + [
            f"{c}_{e}" for e in emotion_labels for c in ("a", "b")
        ]
        save_table(table[matrix_columns], out_path("tables", "p3", 15, "emotion_distribution_matrix"), parquet=False)

        banner("Emotion distributions by domain")
        domains = table[table["level"] == "domain"]
        header = "".join(f"{e[:7]:>9}" for e in emotion_labels)
        item(f"{'domain':<12}{'corpus':<8}{header}")
        for _, row in domains.iterrows():
            for corpus in ("a", "b"):
                values = "".join(f"{row[f'{corpus}_{e}']:>9.3f}" for e in emotion_labels)
                item(f"{row['target'] if corpus == 'a' else '':<12}{corpus.upper():<8}{values}")

        banner("Statistical comparison (post-level, corrected)")
        item(f"{'level':<8}{'target':<16}{'nA':>6}{'nB':>6}{'JS':>7}{'JS p':>8}{'test':>18}{'hard p':>8}{'JS q':>8}")
        for _, row in table.iterrows():
            item(
                f"{row['level']:<8}{str(row['target'])[:15]:<16}{row['n_posts_a']:>6}{row['n_posts_b']:>6}"
                f"{row['js_distance']:>7.3f}{row['js_p']:>8.4f}{row['test']:>18}"
                f"{row['p_value']:>8.4f}{row['fdr_q']:>8.4f}"
            )

    summary = {
        "model": model_name,
        "emotion_labels": emotion_labels,
        "author_requested_labels": author_labels,
        "extra_labels": [e for e in emotion_labels if e not in author_labels],
        "n_windows": int(len(scored)),
        "n_inference_windows_after_author_and_polysemy_filter": int(len(scored_for_inference)),
        "n_authors_retained": int(len(post_metadata)),
        "window_size": window_size,
        "absa_note": (
            "Window proxy for ABSA: emotion is classified over the anchor's local context "
            "and the anchor is treated as the aspect. A window can carry an emotion the "
            "aspect does not own."
        ),
        "statistical_notes": [
            "One deterministic post per author enters inference; windows are then aggregated to post level.",
            "Polysemous anchors are excluded from inferential anchor and strict-domain estimates.",
            "Domain aggregation collapses all unambiguous anchors within a post so each post contributes once per domain.",
            f"Post-level hard-label chi-square only where every expected cell >= {min_expected}; otherwise fixed-margin Monte-Carlo chi-square.",
            "Tier 1 enters inference; Tier 2 retains descriptive effects and nominal bootstrap intervals; Tier 3 is omitted.",
            "Soft distributions are compared by Jensen-Shannon under word-count-decile-stratified post-label permutation.",
            "Benjamini-Hochberg FDR is applied once across all Tier 1 Jensen-Shannon tests in family F2.",
        ],
        "n_results": int(len(table)),
    }
    save_summary(summary, out_path("summaries", "p3", 15, "emotion", ext="json"))
    banner("Done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
