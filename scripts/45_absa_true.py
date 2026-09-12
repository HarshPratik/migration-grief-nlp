"""Phase 3 follow-up - TRUE aspect-based sentiment analysis (ABSA).

Closes the documented gap in scripts/15_emotion_absa.py: that script classifies
the EMOTION of an anchor's local context window (a window proxy). This script
runs a dedicated pre-trained ABSA model instead, where the anchor is passed to
the model as an explicit aspect argument:

    input = "[CLS] <sentence containing the anchor> [SEP] <anchor term> [SEP]"

Model: yangheng/deberta-v3-base-absa-v1.1 (DeBERTa-v3-base fine-tuned for
aspect-based sentiment; 3 classes: negative / neutral / positive). The model
is asked how the author feels ABOUT the anchor, not what emotion the sentence
carries - "I was furious that I missed her phone call" is negative about the
situation but the model scores the polarity directed at "phone call".

Input text differs from script 15 by design: ABSA models are trained on raw
natural language, so this script works on the RAW post `text` (not the
lemmatized, stop-word-stripped window_text). For each (post, anchor) pair the
first sentence containing any surface form of the anchor is extracted.

Statistical protocol mirrors 15_emotion_absa.py exactly so the two analyses
are comparable:
  - one deterministic post per author enters inference
  - polysemous anchors excluded from inferential estimates
  - post-level aggregation (each post contributes one vector per domain)
  - hard-label chi-square / fixed-margin Monte-Carlo chi-square
  - soft distributions compared by Jensen-Shannon distance under
    word-count-decile-stratified permutation, BH-FDR over Tier 1 domains
  - stratified bootstrap CIs per polarity class

Example:
    .venv-analysis\\Scripts\\python.exe scripts/45_absa_true.py --limit 500
    .venv-analysis\\Scripts\\python.exe scripts/45_absa_true.py
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analysis_config import load_analysis_config, load_anchors  # noqa: E402
from analysis_io import (  # noqa: E402
    banner, item, make_rng, out_path, save_summary, save_table,
)

CORPORA = ("a", "b")
MODEL_NAME = "yangheng/deberta-v3-base-absa-v1.1"

# Regex sentence splitter - avoids a hard spaCy model dependency. Good enough
# for Reddit prose; documented as a limitation in the summary.
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


# --------------------------------------------------------------------------
# aspect-sentence extraction
# --------------------------------------------------------------------------


def build_surface_patterns(anchors) -> dict[str, re.Pattern]:
    """term -> compiled whole-word, case-insensitive pattern over surface forms."""
    patterns = {}
    for anchor in anchors.anchors:
        forms = sorted(anchor.surface_forms, key=len, reverse=True)
        body = "|".join(re.escape(f) for f in forms)
        patterns[anchor.term] = re.compile(rf"\b(?:{body})\b", re.IGNORECASE)
    return patterns


def extract_aspect_sentences(
    df: pd.DataFrame, patterns: dict[str, re.Pattern], term_to_meta: dict[str, tuple[str, bool]]
) -> pd.DataFrame:
    """One row per (post_id, anchor): first sentence containing the anchor.

    Returns columns: post_id, corpus, anchor, domain, polysemous, sentence.
    """
    rows = []
    for record in df[["post_id", "corpus", "text"]].itertuples(index=False):
        text = str(record.text)
        if not text.strip():
            continue
        sentences = SENTENCE_SPLIT.split(text.replace("\n", " "))
        for term, pattern in patterns.items():
            domain, polysemous = term_to_meta[term]
            hit_sentence = None
            for sentence in sentences:
                if pattern.search(sentence):
                    hit_sentence = sentence.strip()
                    break
            if hit_sentence:
                rows.append((
                    record.post_id, record.corpus, term, domain, polysemous,
                    hit_sentence[:512],
                ))
    return pd.DataFrame(
        rows,
        columns=["post_id", "corpus", "anchor", "domain", "polysemous", "sentence"],
    )


# --------------------------------------------------------------------------
# ABSA inference
# --------------------------------------------------------------------------


def classify_absa(
    sentences: list[str], aspects: list[str], model_name: str, batch_size: int
) -> tuple[np.ndarray, list[str]]:
    """(n, 3) polarity probability matrix plus normalized label order."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    raw_labels = [model.config.id2label[i] for i in range(model.config.num_labels)]
    labels = [str(x).lower() for x in raw_labels]
    item(f"ABSA device   {device}")
    item(f"ABSA labels   {labels}")

    out = np.zeros((len(sentences), len(labels)), dtype=np.float32)
    for start in range(0, len(sentences), batch_size):
        s_chunk = sentences[start : start + batch_size]
        a_chunk = aspects[start : start + batch_size]
        encoded = tokenizer(
            s_chunk, text_pair=a_chunk, return_tensors="pt",
            padding=True, truncation=True, max_length=192,
        ).to(device)
        with torch.no_grad():
            logits = model(**encoded).logits
        out[start : start + len(s_chunk)] = (
            torch.softmax(logits, dim=-1).float().cpu().numpy()
        )
        if start and start % (batch_size * 50) == 0:
            item(f"classified    {start}/{len(sentences)}", indent=2)
    return out, labels


# --------------------------------------------------------------------------
# statistics (identical protocol to 15_emotion_absa.py)
# --------------------------------------------------------------------------


def jensen_shannon(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p / max(p.sum(), 1e-12)
    q = q / max(q.sum(), 1e-12)
    m = 0.5 * (p + q)

    def _kl(x, y):
        mask = x > 0
        return float(np.sum(x[mask] * np.log2(x[mask] / np.maximum(y[mask], 1e-12))))

    return float(np.sqrt(max(0.5 * _kl(p, m) + 0.5 * _kl(q, m), 0.0)))


def jensen_shannon_rows(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Row-wise JS distance for (n_perm, k) mean matrices."""
    p = p / np.maximum(p.sum(axis=1, keepdims=True), 1e-12)
    q = q / np.maximum(q.sum(axis=1, keepdims=True), 1e-12)
    m = 0.5 * (p + q)

    def _kl(x, y):
        out = np.zeros(x.shape[0])
        mask = x > 0
        terms = np.where(mask, x * np.log2(x / np.maximum(y, 1e-12)), 0.0)
        out = terms.sum(axis=1)
        return out

    return np.sqrt(np.maximum(0.5 * _kl(p, m) + 0.5 * _kl(q, m), 0.0))


def stratified_permutation_null(
    pooled: np.ndarray, group_id: np.ndarray, strata: np.ndarray,
    *, n_permutations: int, rng, chunk: int = 2000,
) -> np.ndarray:
    """Vectorized JS null: labels permuted within word-count strata.

    Identical semantics to the scalar loop used in 15_emotion_absa.py, but
    batched over permutations so 10k draws finish in seconds.
    """
    null = np.empty(n_permutations)
    n = len(pooled)
    pooled32 = np.asarray(pooled, dtype=np.float32)
    for lo in range(0, n_permutations, chunk):
        hi = min(lo + chunk, n_permutations)
        perm = np.broadcast_to(group_id, (hi - lo, n)).copy()
        for stratum in np.unique(strata):
            idx = np.flatnonzero(strata == stratum)
            keys = rng.random((hi - lo, len(idx)))
            order = np.argsort(keys, axis=1)
            perm[:, idx] = np.take_along_axis(perm[:, idx], order, axis=1)
        mask0 = perm == 0
        mean0 = (mask0.astype(np.float32) @ pooled32) / mask0.sum(axis=1, keepdims=True)
        mask1 = ~mask0
        mean1 = (mask1.astype(np.float32) @ pooled32) / mask1.sum(axis=1, keepdims=True)
        null[lo:hi] = jensen_shannon_rows(mean0, mean1)
    return null


def contingency_test(table: np.ndarray, *, min_expected: int, rng, n_permutations: int) -> dict:
    from scipy.stats import chi2_contingency

    counts = np.asarray(table, dtype=int)
    counts = counts[:, counts.sum(axis=0) > 0]
    if counts.shape[1] < 2 or counts.sum() == 0:
        return {"test": "none", "statistic": np.nan, "p_value": np.nan, "min_expected": 0.0}

    stat, p, _, expected = chi2_contingency(counts, correction=False)
    smallest = float(expected.min())
    if smallest >= min_expected:
        return {"test": "chi2", "statistic": float(stat), "p_value": float(p), "min_expected": smallest}

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


def stratified_bootstrap_means(values, word_count, *, n_bootstrap, rng) -> np.ndarray:
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
    parser = argparse.ArgumentParser(description="True ABSA over anchor aspect-sentences")
    parser.add_argument("--corpus", choices=["a", "b", "both"], default="both")
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=0, help="classify only the first N pairs (debug)")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = load_analysis_config(seed=args.seed)
    anchors = load_anchors()
    analysis_dir = cfg.path("analysis_dir")
    min_expected = int(cfg.get("stats.min_expected_cell"))
    n_permutations = int(cfg.get("stats.n_permutations"))
    n_boot = int(cfg.get("stats.n_bootstrap"))
    rng = make_rng(cfg.seed)

    tier_path = out_path("tables", "p2", 9, "anchor_coverage")
    if not tier_path.exists():
        raise SystemExit(f"missing {tier_path}; run scripts/09_anchor_coverage.py first")
    tier_table = pd.read_csv(tier_path)
    anchor_tier = dict(zip(tier_table["anchor"].astype(str), tier_table["tier_observed"].astype(int)))

    term_to_meta = {
        a.term: (domain_name, a.polysemous)
        for domain_name, domain in anchors.domains.items()
        for a in domain.anchors
    }
    polysemous = {a.term for a in anchors.anchors if a.polysemous}
    patterns = build_surface_patterns(anchors)

    labels_to_run = ["a", "b"] if args.corpus == "both" else [args.corpus]
    # Filesystem-safe model tag ("deberta-v3-base-absa-v1.1" would lose its
    # ".1" to save_table's suffix handling).
    model_tag = args.model.split("/")[-1].replace(".", "_")

    banner("Aspect-sentence extraction (raw text)")
    scored_frames = []
    polarity_labels: list[str] | None = None
    for label in labels_to_run:
        cache_path = analysis_dir / f"absa_{label}_{model_tag}.parquet"
        if cache_path.exists() and not args.limit:
            cached = pd.read_parquet(cache_path)
            item(f"corpus {label}: loaded {len(cached)} cached scored pairs from {cache_path.name}")
            if polarity_labels is None:
                polarity_labels = [c[2:] for c in cached.columns if c.startswith("p_")]
            scored_frames.append(cached)
            continue

        corpus = pd.read_parquet(
            analysis_dir / f"corpus_{label}.parquet", columns=["post_id", "text"]
        ).assign(corpus=label)
        pairs = extract_aspect_sentences(corpus, patterns, term_to_meta)
        item(f"corpus {label}: {len(pairs)} (post, anchor) pairs "
             f"over {pairs['post_id'].nunique()} posts, {pairs['anchor'].nunique()} anchors")
        if args.limit:
            pairs = pairs.head(args.limit)

        banner(f"ABSA classification (corpus {label})")
        item(f"model         {args.model}")
        probabilities, polarity_labels = classify_absa(
            pairs["sentence"].tolist(),
            pairs["anchor"].tolist(),
            args.model,
            args.batch_size,
        )
        scored_one = pairs.copy()
        for i, name in enumerate(polarity_labels):
            scored_one[f"p_{name}"] = probabilities[:, i]
        scored_one["polarity"] = [polarity_labels[i] for i in probabilities.argmax(axis=1)]
        scored_frames.append(scored_one)

        if args.limit:
            item("debug limit active: classification validated; no canonical outputs written")
            print(scored_one[["anchor", "sentence", "polarity"]].head(10).to_string())
            return 0

        save_table(scored_one, cache_path.with_suffix(""), parquet=True, csv=False)

    scored = pd.concat(scored_frames, ignore_index=True)
    probability_columns = [f"p_{name}" for name in polarity_labels]

    # ---- author cap + polysemy filter (same as 15) ------------------------
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

    probability_columns = [f"p_{name}" for name in polarity_labels]
    per_post = {
        "domain": (
            scored_for_inference.groupby(["corpus", "domain", "post_id"])[probability_columns]
            .mean().reset_index()
            .merge(post_metadata[["corpus", "post_id", "word_count"]], on=["corpus", "post_id"])
        ),
        "anchor": (
            scored_for_inference.groupby(["corpus", "anchor", "post_id"])[probability_columns]
            .mean().reset_index()
            .merge(post_metadata[["corpus", "post_id", "word_count"]], on=["corpus", "post_id"])
        ),
    }

    # ---- per-domain and per-anchor comparison ------------------------------
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

            counts = np.vstack([
                np.bincount(vectors_a.argmax(axis=1), minlength=len(polarity_labels)),
                np.bincount(vectors_b.argmax(axis=1), minlength=len(polarity_labels)),
            ])
            test = (
                contingency_test(counts, min_expected=min_expected, rng=rng,
                                 n_permutations=n_permutations)
                if inferential
                else {"test": "descriptive_only", "statistic": np.nan,
                      "p_value": np.nan, "min_expected": np.nan}
            )

            observed_js = jensen_shannon(mean_a, mean_b)
            pooled = np.vstack([vectors_a, vectors_b])
            lengths = np.concatenate([a["word_count"].to_numpy(), b["word_count"].to_numpy()])
            strata = pd.qcut(lengths, 10, labels=False, duplicates="drop")
            null: np.ndarray | list = []
            if inferential:
                observed_groups = np.concatenate(
                    [np.zeros(len(a), dtype=np.int8), np.ones(len(b), dtype=np.int8)]
                )
                null = stratified_permutation_null(
                    pooled, observed_groups, np.asarray(strata),
                    n_permutations=n_permutations, rng=rng,
                )
            js_p = (
                float((np.sum(np.asarray(null) >= observed_js) + 1) / (len(null) + 1))
                if len(null) else np.nan
            )

            bootstrap_a = stratified_bootstrap_means(
                vectors_a, a["word_count"].to_numpy(), n_bootstrap=n_boot, rng=rng)
            bootstrap_b = stratified_bootstrap_means(
                vectors_b, b["word_count"].to_numpy(), n_bootstrap=n_boot, rng=rng)
            record = {
                "level": level, "target": name, "tier": tier,
                "in_fdr_family": inferential,
                "n_posts_a": len(a), "n_posts_b": len(b),
                "n_pairs_a": int(len(scored_for_inference[
                    (scored_for_inference[key] == name) & (scored_for_inference["corpus"] == "a")])),
                "n_pairs_b": int(len(scored_for_inference[
                    (scored_for_inference[key] == name) & (scored_for_inference["corpus"] == "b")])),
                "js_distance": round(observed_js, 4), "js_p": round(js_p, 5),
                "test": test["test"], "statistic": round(test["statistic"], 3),
                "p_value": round(test["p_value"], 5),
                "min_expected_cell": round(test["min_expected"], 2),
            }
            for i, pol in enumerate(polarity_labels):
                record[f"a_{pol}"] = round(float(mean_a[i]), 4)
                record[f"b_{pol}"] = round(float(mean_b[i]), 4)
                draws = bootstrap_a[:, i] - bootstrap_b[:, i]
                low, high = np.percentile(draws, [2.5, 97.5])
                record[f"diff_{pol}"] = round(float(mean_a[i] - mean_b[i]), 4)
                record[f"ci_{pol}"] = f"[{low:+.3f}, {high:+.3f}]"
            results.append(record)

    table = pd.DataFrame(results)
    if table.empty:
        raise SystemExit("no comparisons produced - check inputs")

    table["fdr_q"] = np.nan
    family = table["in_fdr_family"]
    table.loc[family, "fdr_q"] = benjamini_hochberg(table.loc[family, "js_p"].to_numpy())
    table = table.sort_values(["level", "js_distance"], ascending=[True, False])
    save_table(table, out_path("tables", "p3", 45, "absa_comparison"), parquet=False)

    matrix_columns = ["level", "target", "n_pairs_a", "n_pairs_b"] + [
        f"{c}_{p}" for p in polarity_labels for c in ("a", "b")
    ]
    save_table(table[matrix_columns], out_path("tables", "p3", 45, "absa_distribution_matrix"), parquet=False)

    banner("ABSA polarity distributions by domain")
    domains = table[table["level"] == "domain"]
    header = "".join(f"{p[:9]:>11}" for p in polarity_labels)
    item(f"{'domain':<12}{'corpus':<8}{header}")
    for _, row in domains.iterrows():
        for corpus in ("a", "b"):
            values = "".join(f"{row[f'{corpus}_{p}']:>11.3f}" for p in polarity_labels)
            item(f"{row['target'] if corpus == 'a' else '':<12}{corpus.upper():<8}{values}")

    banner("Statistical comparison (post-level, corrected)")
    item(f"{'level':<8}{'target':<16}{'nA':>6}{'nB':>6}{'JS':>7}{'JS p':>8}{'test':>18}{'JS q':>8}")
    for _, row in table.iterrows():
        item(
            f"{row['level']:<8}{str(row['target'])[:15]:<16}{row['n_posts_a']:>6}{row['n_posts_b']:>6}"
            f"{row['js_distance']:>7.3f}{row['js_p']:>8.4f}{row['test']:>18}{row['fdr_q']:>8.4f}"
        )

    summary = {
        "model": args.model,
        "polarity_labels": polarity_labels,
        "n_pairs": int(len(scored)),
        "n_inference_pairs_after_author_and_polysemy_filter": int(len(scored_for_inference)),
        "unit": "one (post, anchor) pair; first raw sentence containing the anchor",
        "difference_from_script_15": (
            "15 classifies window EMOTION (7 classes, lemmatized window_text, aspect implicit); "
            "45 runs a dedicated ABSA model with the anchor as an explicit aspect argument "
            "on RAW sentences (3-class polarity directed at the aspect)."
        ),
        "limitations": [
            "Regex sentence splitting (no spaCy dependency); Reddit prose with unusual punctuation may mis-split.",
            "First mention per (post, anchor) only; later mentions of the same anchor in a post are not scored.",
            "DeBERTa-v3 ABSA model was fine-tuned largely on review-style data; transfer to grief/migration narrative is an assumption.",
        ],
        "statistical_notes": [
            "Identical protocol to 15_emotion_absa.py: author cap, polysemy exclusion, post-level aggregation, "
            "chi-square/MC on hard labels, JS distance under word-count-decile-stratified permutation, "
            "BH-FDR across Tier 1 domain tests.",
        ],
        "n_results": int(len(table)),
    }
    save_summary(summary, out_path("summaries", "p3", 45, "absa", ext="json"))
    banner("Done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
