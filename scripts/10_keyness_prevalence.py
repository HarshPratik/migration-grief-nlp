"""Phase 2 - keyness and prevalence: the frequency comparison of Corpus A vs B.

This is the analysis this corpus size actually supports. Measured split-half
reliability puts corpus-trained neighbour lists at chance below f=200, so the
embedding results elsewhere in this project are weak. Counting is not: a word
that occurs 241 times in one corpus and 0 times in the other is a fact about
the data, not an artefact of an estimator, and keyness is the standard way to
report it.

Three analyses, all on `tokens_lemma` (stopwords removed, multi-word anchors
already collapsed to single tokens by script 08):

(a) WHOLE-VOCABULARY KEYNESS. Log-likelihood (Dunning 1993 G2) for
    significance, Log Ratio (Hardie 2014) for effect size, %DIFF
    (Gabrielatos & Marchi 2012) as the descriptive third view, and a
    Benjamini-Hochberg q-value over the whole tested vocabulary (family F4).
    Raw counts are NEVER compared: the corpora are 123k vs 109k tokens, so
    every table carries per-10,000-token normalised rates beside the counts.

(b) ANCHOR AND DOMAIN KEYNESS. The same statistics restricted to the 71 seed
    anchors and to the 4 domains pooled. This is where the strongest anchor
    findings live, because many anchors are corpus-SPECIFIC (`funeral` 0 in A
    / 241 in B; `culture_shock` 27 in A / 0 in B). No similarity measure can
    say anything about a word with a zero vector in one corpus. Keyness
    handles it natively - a zero cell is the maximum possible effect, not a
    missing value.

(c) PREVALENCE CONTRASTS. Keyness counts tokens, and tokens inside one post
    are not independent observations: one bereaved user writing `funeral` 30
    times can carry a G2 on their own. Prevalence counts POSTS instead - the
    unit that actually varies - and tests it with Fisher's exact test plus an
    odds ratio and a 95% CI (analytic, from the log-OR standard error, with a
    Haldane-Anscombe 0.5 correction when any cell is zero). Where (a)/(b) and
    (c) disagree, (c) is the safer claim.

Honest limits, stated because a misleading result is worse than a null one:
  * G2 p-values assume token independence and this data violates it. Treat the
    q-values as a ranking device, not as calibrated error rates; the
    `top_post_share_*` and `df_*` columns are there so a reader can see when a
    keyword is really one post shouting.
  * Log Ratio with a zero cell is bounded only by the adjustment constant, so
    it is reported alongside an explicit `zero_cell` flag and a second,
    smoothed variant (Kilgarriff's simple maths) that cannot run away.
  * `strict` domain rows drop anchors flagged polysemous in anchors.yaml. The
    Digital domain's raw total is dominated by `connection` / `disconnect`,
    which concordance shows are usually interpersonal. Both are reported.

Examples:
    D:\\research_paper\\.venv-analysis\\Scripts\\python.exe scripts/10_keyness_prevalence.py
    ... scripts/10_keyness_prevalence.py --min-freq 20
    ... scripts/10_keyness_prevalence.py --top 25 --simple-maths-k 0.5
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analysis_config import load_analysis_config, load_anchors  # noqa: E402
from analysis_io import as_token_list, banner, item, out_path, save_summary, save_table  # noqa: E402

NORM_PER = 10_000          # every rate in every table is per 10k running tokens
ZERO_ADJUST = 0.5          # Hardie's zero-cell correction, applied to raw counts
DEFAULT_SIMPLE_MATHS_K = 1.0   # Kilgarriff simple maths, added to per-10k rates


# --------------------------------------------------------------------------
# counting
# --------------------------------------------------------------------------


class CorpusCounts:
    """Frequencies for one corpus. Built once, never mutated afterwards."""

    __slots__ = ("label", "freq", "docs", "top_post", "n_tokens", "n_posts")

    def __init__(self, label: str, freq: Counter, docs: Counter,
                 top_post: Mapping[str, int], n_tokens: int, n_posts: int) -> None:
        self.label = label
        self.freq = freq
        self.docs = docs
        self.top_post = dict(top_post)
        self.n_tokens = n_tokens
        self.n_posts = n_posts

    def rate(self, word: str) -> float:
        return NORM_PER * self.freq.get(word, 0) / max(self.n_tokens, 1)


def count_corpus(df: pd.DataFrame, label: str, column: str) -> CorpusCounts:
    """Token frequency, document frequency, and single-post concentration.

    `top_post` is the largest number of occurrences of a word inside any one
    post. It is the cheapest available dispersion warning: a word with
    freq 60 and top_post 41 is one post, not a corpus pattern.
    """
    freq: Counter = Counter()
    docs: Counter = Counter()
    top_post: dict[str, int] = {}
    total = 0
    for cell in df[column]:
        tokens = as_token_list(cell)
        total += len(tokens)
        counts = Counter(tokens)
        freq.update(counts)
        docs.update(counts.keys())
        for word, n in counts.items():
            if n > top_post.get(word, 0):
                top_post[word] = n
    return CorpusCounts(label, freq, docs, top_post, total, len(df))


def doc_sets(df: pd.DataFrame, column: str) -> list[set[str]]:
    """One set of distinct tokens per post - the unit prevalence counts over."""
    return [set(as_token_list(cell)) for cell in df[column]]


# --------------------------------------------------------------------------
# statistics (pure functions, no I/O)
# --------------------------------------------------------------------------


def log_likelihood(f_a: int, f_b: int, n_a: int, n_b: int) -> float:
    """Dunning (1993) G2 for a 2x2 word/corpus table. Always >= 0.

    Direction is not encoded here; callers read it off the normalised rates.

    >>> round(log_likelihood(100, 10, 100_000, 100_000), 2)
    89.42
    """
    total_f = f_a + f_b
    total_n = n_a + n_b
    if total_f == 0 or total_n == 0:
        return 0.0
    e_a = n_a * total_f / total_n
    e_b = n_b * total_f / total_n
    g = 0.0
    if f_a > 0 and e_a > 0:
        g += f_a * math.log(f_a / e_a)
    if f_b > 0 and e_b > 0:
        g += f_b * math.log(f_b / e_b)
    return 2.0 * g


def log_ratio(f_a: int, f_b: int, n_a: int, n_b: int, *, zero_adjust: float = ZERO_ADJUST) -> float:
    """Hardie (2014) Log Ratio: log2 of the ratio of normalised frequencies.

    A zero cell makes the ratio undefined, so a zero RAW count is replaced by
    `zero_adjust` (0.5 - half the smallest observable count). The resulting
    value is therefore bounded by the corpus size, not by the data, which is
    why every caller also carries the `zero_cell` flag.
    """
    a = f_a if f_a > 0 else zero_adjust
    b = f_b if f_b > 0 else zero_adjust
    return math.log2((a / max(n_a, 1)) / (b / max(n_b, 1)))


def log_ratio_simple_maths(rate_a: float, rate_b: float, k: float) -> float:
    """Kilgarriff's simple maths on per-10k rates: log2((ra+k)/(rb+k)).

    Adding k to BOTH sides shrinks the ratio for rare words, so a 1-vs-0
    difference can no longer outrank a 200-vs-3 difference. Reported beside
    the unsmoothed Log Ratio as a sanity rail.
    """
    return math.log2((rate_a + k) / (rate_b + k))


def percent_diff(rate_a: float, rate_b: float) -> float:
    """%DIFF (Gabrielatos & Marchi 2012). Infinite when the B rate is zero."""
    if rate_b == 0:
        return math.inf if rate_a > 0 else 0.0
    return 100.0 * (rate_a - rate_b) / rate_b


def benjamini_hochberg(pvalues: Sequence[float]) -> list[float]:
    """BH step-up q-values, order preserved. Monotone by construction.

    >>> [round(q, 4) for q in benjamini_hochberg([0.01, 0.02, 0.5])]
    [0.03, 0.03, 0.5]
    """
    n = len(pvalues)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: pvalues[i])
    out = [1.0] * n
    running = 1.0
    for rank_from_top, index in enumerate(reversed(order), start=1):
        rank = n - rank_from_top + 1
        running = min(running, pvalues[index] * n / rank)
        out[index] = min(1.0, running)
    return out


def odds_ratio_ci(
    with_a: int, without_a: int, with_b: int, without_b: int, *, alpha: float = 0.05
) -> tuple[float, float, float, bool]:
    """Analytic CI for the odds ratio via the log-OR standard error.

    Returns (or_used, ci_low, ci_high, haldane_corrected). A Haldane-Anscombe
    0.5 is added to EVERY cell when ANY cell is zero, because the standard
    error 1/a+1/b+1/c+1/d is undefined otherwise. The corrected OR is returned
    alongside so the CI and its point estimate are always consistent; the
    uncorrected sample OR from `fisher_exact` is reported in its own column.
    """
    cells = [with_a, without_a, with_b, without_b]
    corrected = min(cells) == 0
    a, b, c, d = (float(x) + 0.5 for x in cells) if corrected else (float(x) for x in cells)
    if b == 0 or c == 0:
        return math.nan, math.nan, math.nan, corrected
    ratio = (a * d) / (b * c)
    se = math.sqrt(1.0 / a + 1.0 / b + 1.0 / c + 1.0 / d)
    z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    return ratio, math.exp(math.log(ratio) - z * se), math.exp(math.log(ratio) + z * se), corrected


# --------------------------------------------------------------------------
# row builders
# --------------------------------------------------------------------------


def keyness_row(word: str, a: CorpusCounts, b: CorpusCounts, *, k: float) -> dict[str, Any]:
    """One keyness record. Counts, rates, effect sizes and a dispersion warning."""
    f_a, f_b = a.freq.get(word, 0), b.freq.get(word, 0)
    r_a, r_b = a.rate(word), b.rate(word)
    g2 = log_likelihood(f_a, f_b, a.n_tokens, b.n_tokens)
    share_a = (a.top_post.get(word, 0) / f_a) if f_a else 0.0
    share_b = (b.top_post.get(word, 0) / f_b) if f_b else 0.0
    return {
        "word": word,
        "freq_a": f_a,
        "freq_b": f_b,
        "rate_a_per10k": r_a,
        "rate_b_per10k": r_b,
        "df_a": a.docs.get(word, 0),
        "df_b": b.docs.get(word, 0),
        "pct_posts_a": 100.0 * a.docs.get(word, 0) / max(a.n_posts, 1),
        "pct_posts_b": 100.0 * b.docs.get(word, 0) / max(b.n_posts, 1),
        "g2": g2,
        "p_value": float(stats.chi2.sf(g2, 1)) if g2 > 0 else 1.0,
        "log_ratio": log_ratio(f_a, f_b, a.n_tokens, b.n_tokens),
        "log_ratio_sm": log_ratio_simple_maths(r_a, r_b, k),
        "pct_diff": percent_diff(r_a, r_b),
        "direction": "A" if r_a > r_b else ("B" if r_b > r_a else "none"),
        "zero_cell": (f_a == 0) or (f_b == 0),
        # Largest share of the word's tokens coming from a single post. High
        # values mean the G2 rests on one author, not on a corpus pattern.
        # `_key` is the share in the corpus the word is KEY TO - the only side
        # that can inflate the statistic. A word key to A with share_b = 1.0 is
        # just rare in B, which is the finding, not a fragility.
        "top_post_share_a": share_a,
        "top_post_share_b": share_b,
        "top_post_share_key": share_a if r_a > r_b else share_b,
    }


def pooled_keyness_row(
    name: str, terms: Iterable[str], a: CorpusCounts, b: CorpusCounts, *, k: float
) -> dict[str, Any]:
    """Keyness for a set of terms counted as one pooled category."""
    terms = tuple(terms)
    f_a = sum(a.freq.get(t, 0) for t in terms)
    f_b = sum(b.freq.get(t, 0) for t in terms)
    r_a = NORM_PER * f_a / max(a.n_tokens, 1)
    r_b = NORM_PER * f_b / max(b.n_tokens, 1)
    g2 = log_likelihood(f_a, f_b, a.n_tokens, b.n_tokens)
    return {
        "name": name,
        "n_terms": len(terms),
        "freq_a": f_a,
        "freq_b": f_b,
        "rate_a_per10k": r_a,
        "rate_b_per10k": r_b,
        "g2": g2,
        "p_value": float(stats.chi2.sf(g2, 1)) if g2 > 0 else 1.0,
        "log_ratio": log_ratio(f_a, f_b, a.n_tokens, b.n_tokens),
        "log_ratio_sm": log_ratio_simple_maths(r_a, r_b, k),
        "pct_diff": percent_diff(r_a, r_b),
        "direction": "A" if r_a > r_b else ("B" if r_b > r_a else "none"),
        "zero_cell": (f_a == 0) or (f_b == 0),
    }


def prevalence_row(
    level: str,
    name: str,
    terms: Iterable[str],
    sets_a: Sequence[set[str]],
    sets_b: Sequence[set[str]],
) -> dict[str, Any]:
    """Posts containing ANY of `terms`, contrasted with Fisher's exact test."""
    wanted = set(terms)
    with_a = sum(1 for s in sets_a if wanted & s)
    with_b = sum(1 for s in sets_b if wanted & s)
    without_a = len(sets_a) - with_a
    without_b = len(sets_b) - with_b

    table = [[with_a, without_a], [with_b, without_b]]
    sample_or, p_value = stats.fisher_exact(table, alternative="two-sided")
    ci_or, ci_low, ci_high, corrected = odds_ratio_ci(with_a, without_a, with_b, without_b)

    return {
        "level": level,
        "name": name,
        "n_terms": len(wanted),
        "posts_with_a": with_a,
        "posts_without_a": without_a,
        "posts_with_b": with_b,
        "posts_without_b": without_b,
        "n_posts_a": len(sets_a),
        "n_posts_b": len(sets_b),
        "pct_posts_a": 100.0 * with_a / max(len(sets_a), 1),
        "pct_posts_b": 100.0 * with_b / max(len(sets_b), 1),
        "fisher_p": float(p_value),
        "odds_ratio_sample": float(sample_or),
        "odds_ratio_ci": ci_or,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "haldane_corrected": corrected,
    }


def attach_fdr(rows: list[dict[str, Any]], family: str, *, tested: Sequence[bool] | None = None,
               p_key: str = "p_value", q_key: str = "q_value") -> list[dict[str, Any]]:
    """Return NEW rows carrying a BH q-value. Untested rows get q = None.

    A row is 'untested' when the category is empty in BOTH corpora: including
    it would inflate the family size with tests that could never reject.
    """
    flags = list(tested) if tested is not None else [True] * len(rows)
    indices = [i for i, ok in enumerate(flags) if ok]
    qvalues = benjamini_hochberg([rows[i][p_key] for i in indices])
    lookup = dict(zip(indices, qvalues))
    return [
        {**row, q_key: lookup.get(i), "fdr_family": family, "fdr_n_tests": len(indices)}
        for i, row in enumerate(rows)
    ]


# --------------------------------------------------------------------------
# reporting helpers
# --------------------------------------------------------------------------


def print_keyness_block(title: str, frame: pd.DataFrame, top: int) -> None:
    banner(title)
    item(f"{'word':<22}{'fA':>6}{'fB':>6}{'/10kA':>8}{'/10kB':>8}{'G2':>9}{'LR':>7}{'q':>10}{'dfA/dfB':>10}")
    for _, row in frame.head(top).iterrows():
        q = row["q_value"]
        q_txt = "n/a" if q is None or pd.isna(q) else f"{q:.2e}"
        item(
            f"{str(row['word'])[:21]:<22}{row['freq_a']:>6}{row['freq_b']:>6}"
            f"{row['rate_a_per10k']:>8.1f}{row['rate_b_per10k']:>8.1f}"
            f"{row['g2']:>9.1f}{row['log_ratio']:>7.2f}{q_txt:>10}"
            f"{str(row['df_a']) + '/' + str(row['df_b']):>10}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Keyness (G2 / Log Ratio / %DIFF) and post-level prevalence, Corpus A vs B"
    )
    parser.add_argument("--analysis-dir", type=Path, default=None)
    parser.add_argument("--column", default="tokens_lemma",
                        help="token column to count (default: tokens_lemma)")
    parser.add_argument("--min-freq", type=int, default=10,
                        help="minimum COMBINED frequency for a word to enter the keyness family")
    parser.add_argument("--top", type=int, default=15, help="rows printed per direction")
    parser.add_argument("--simple-maths-k", type=float, default=DEFAULT_SIMPLE_MATHS_K,
                        help="constant added to both per-10k rates for the smoothed Log Ratio")
    parser.add_argument("--alpha", type=float, default=0.05, help="CI level for the odds ratios")
    args = parser.parse_args()

    if args.min_freq < 1:
        raise SystemExit("--min-freq must be >= 1")
    if args.simple_maths_k <= 0:
        raise SystemExit("--simple-maths-k must be > 0 (it exists to bound the ratio)")

    cfg = load_analysis_config()
    anchors = load_anchors()
    analysis_dir = args.analysis_dir or cfg.path("analysis_dir")
    fdr_q = float(cfg.get("stats.fdr_q", 0.05))

    frames: dict[str, pd.DataFrame] = {}
    for label in ("a", "b"):
        path = analysis_dir / f"corpus_{label}.parquet"
        if not path.exists():
            raise SystemExit(f"missing {path}; run scripts/08_reprocess_tokens.py first")
        frames[label] = pd.read_parquet(path, columns=[args.column])

    counts = {label: count_corpus(frames[label], label, args.column) for label in ("a", "b")}
    a, b = counts["a"], counts["b"]
    sets_a, sets_b = doc_sets(frames["a"], args.column), doc_sets(frames["b"], args.column)

    banner("Corpus sizes (rates below are per 10,000 running tokens)")
    item(f"A migration : {a.n_posts:>4} posts, {a.n_tokens:>7} tokens, vocab {len(a.freq):>6}")
    item(f"B grief     : {b.n_posts:>4} posts, {b.n_tokens:>7} tokens, vocab {len(b.freq):>6}")
    item(f"size ratio A/B = {a.n_tokens / max(b.n_tokens, 1):.3f} - raw counts are never compared directly")

    # ---- (a) whole-vocabulary keyness -------------------------------------
    vocabulary = sorted(
        w for w in set(a.freq) | set(b.freq)
        if a.freq.get(w, 0) + b.freq.get(w, 0) >= args.min_freq
    )
    rows = [keyness_row(w, a, b, k=args.simple_maths_k) for w in vocabulary]
    rows = attach_fdr(rows, "F4_keyness_vocabulary")
    keyness = pd.DataFrame(rows).sort_values("g2", ascending=False, kind="mergesort").reset_index(drop=True)

    save_table(keyness, out_path("tables", "p2", 10, "keyness"), parquet=False)
    a_key = keyness[keyness["direction"] == "A"].head(100).reset_index(drop=True)
    b_key = keyness[keyness["direction"] == "B"].head(100).reset_index(drop=True)
    save_table(a_key, out_path("tables", "p2", 10, "keyness_top100", "a"), parquet=False)
    save_table(b_key, out_path("tables", "p2", 10, "keyness_top100", "b"), parquet=False)

    banner("Vocabulary keyness")
    item(f"tested vocabulary: {len(keyness)} words (combined freq >= {args.min_freq})")
    item(f"significant at BH q < {fdr_q}: {int((keyness['q_value'] < fdr_q).sum())}")
    item(f"  of which key to A: {int(((keyness['q_value'] < fdr_q) & (keyness['direction'] == 'A')).sum())}")
    item(f"  of which key to B: {int(((keyness['q_value'] < fdr_q) & (keyness['direction'] == 'B')).sum())}")

    print_keyness_block(f"Top {args.top} words KEY TO CORPUS A (migration)", a_key, args.top)
    print_keyness_block(f"Top {args.top} words KEY TO CORPUS B (grief)", b_key, args.top)

    # ---- (b) anchor and domain keyness ------------------------------------
    anchor_rows = []
    for anchor in anchors.anchors:
        row = keyness_row(anchor.term, a, b, k=args.simple_maths_k)
        anchor_rows.append({
            "domain": anchor.domain,
            "anchor": row.pop("word"),
            "polysemous": anchor.polysemous,
            "multiword": anchor.is_multiword,
            "tier_expected": anchor.tier,
            **row,
            "corpus_specific": (row["freq_a"] == 0) != (row["freq_b"] == 0),
        })
    anchor_rows = attach_fdr(
        anchor_rows, "F4_keyness_anchors",
        tested=[r["freq_a"] + r["freq_b"] > 0 for r in anchor_rows],
    )
    anchor_keyness = (
        pd.DataFrame(anchor_rows)
        .sort_values("g2", ascending=False, kind="mergesort")
        .reset_index(drop=True)
    )
    save_table(anchor_keyness, out_path("tables", "p2", 10, "anchor_keyness"), parquet=False)

    domain_rows: list[dict[str, Any]] = []
    for variant in ("all", "strict"):
        variant_rows = []
        for name, domain in anchors.domains.items():
            terms = (
                domain.terms if variant == "all"
                else tuple(x.term for x in domain.anchors if not x.polysemous)
            )
            row = pooled_keyness_row(name, terms, a, b, k=args.simple_maths_k)
            variant_rows.append({
                "domain": row.pop("name"),
                "label": domain.label,
                "variant": variant,
                **row,
            })
        domain_rows.extend(attach_fdr(variant_rows, f"F4_keyness_domains_{variant}"))
    domain_keyness = pd.DataFrame(domain_rows)
    save_table(domain_keyness, out_path("tables", "p2", 10, "domain_keyness"), parquet=False)

    banner("Domain keyness (pooled anchors; 'strict' drops polysemous anchors)")
    item(f"{'domain':<12}{'var':<8}{'fA':>7}{'fB':>7}{'/10kA':>8}{'/10kB':>8}{'G2':>9}{'LR':>7}{'q':>11}")
    for _, row in domain_keyness.iterrows():
        q = row["q_value"]
        q_txt = "n/a" if q is None or pd.isna(q) else f"{q:.3e}"
        item(
            f"{row['domain']:<12}{row['variant']:<8}{row['freq_a']:>7}{row['freq_b']:>7}"
            f"{row['rate_a_per10k']:>8.1f}{row['rate_b_per10k']:>8.1f}"
            f"{row['g2']:>9.1f}{row['log_ratio']:>7.2f}{q_txt:>11}"
        )

    corpus_specific = anchor_keyness[anchor_keyness["corpus_specific"]]
    banner(f"Corpus-SPECIFIC anchors ({len(corpus_specific)}) - zero in one corpus, present in the other")
    item("No similarity measure can score these. Keyness can.")
    for _, row in corpus_specific.sort_values("g2", ascending=False).iterrows():
        q = row["q_value"]
        q_txt = "n/a" if q is None or pd.isna(q) else f"{q:.1e}"
        item(
            f"{row['domain']:<10}{str(row['anchor'])[:21]:<22}"
            f"A={row['freq_a']:>4} B={row['freq_b']:>4}  G2={row['g2']:>7.1f}  q={q_txt}"
        )

    # ---- (c) prevalence contrasts -----------------------------------------
    prevalence_rows: list[dict[str, Any]] = []

    domain_prevalence: list[dict[str, Any]] = []
    for variant in ("all", "strict"):
        variant_rows = []
        for name, domain in anchors.domains.items():
            terms = (
                domain.terms if variant == "all"
                else tuple(x.term for x in domain.anchors if not x.polysemous)
            )
            row = prevalence_row("domain", name, terms, sets_a, sets_b)
            variant_rows.append({**row, "domain": name, "variant": variant})
        domain_prevalence.extend(attach_fdr(variant_rows, f"F4_prevalence_domains_{variant}",
                                            p_key="fisher_p"))
    prevalence_rows.extend(domain_prevalence)

    anchor_prevalence = [
        {
            **prevalence_row("anchor", anchor.term, (anchor.term,), sets_a, sets_b),
            "domain": anchor.domain,
            "variant": "polysemous" if anchor.polysemous else "strict",
        }
        for anchor in anchors.anchors
    ]
    anchor_prevalence = attach_fdr(
        anchor_prevalence, "F4_prevalence_anchors", p_key="fisher_p",
        tested=[r["posts_with_a"] + r["posts_with_b"] > 0 for r in anchor_prevalence],
    )
    prevalence_rows.extend(anchor_prevalence)

    prevalence = pd.DataFrame(prevalence_rows)
    save_table(prevalence, out_path("tables", "p2", 10, "prevalence"), parquet=False)

    banner("Domain prevalence (posts containing >=1 anchor; Fisher exact, two-sided)")
    item(f"{'domain':<12}{'var':<8}{'postsA':>12}{'postsB':>12}{'OR':>8}{'95% CI':>20}{'p':>11}{'q':>11}")
    for row in domain_prevalence:
        ci = f"[{row['ci_low']:.2f}, {row['ci_high']:.2f}]"
        q = row["q_value"]
        q_txt = "n/a" if q is None else f"{q:.3e}"
        item(
            f"{row['domain']:<12}{row['variant']:<8}"
            f"{row['posts_with_a']:>5}/{row['n_posts_a']:<3}({row['pct_posts_a']:>4.1f}%)"
            f"{row['posts_with_b']:>5}/{row['n_posts_b']:<3}({row['pct_posts_b']:>4.1f}%)"
            f"{row['odds_ratio_ci']:>8.2f}{ci:>20}{row['fisher_p']:>11.2e}{q_txt:>11}"
        )

    digital = next(
        (r for r in domain_prevalence if r["domain"] == "digital" and r["variant"] == "all"), None
    )
    digital_strict = next(
        (r for r in domain_prevalence if r["domain"] == "digital" and r["variant"] == "strict"), None
    )
    if digital is not None:
        banner("HEADLINE - Digital-domain prevalence contrast")
        item(f"A: {digital['posts_with_a']}/{digital['n_posts_a']} posts ({digital['pct_posts_a']:.1f}%)"
             f"   B: {digital['posts_with_b']}/{digital['n_posts_b']} posts ({digital['pct_posts_b']:.1f}%)")
        item(f"Fisher exact two-sided p = {digital['fisher_p']:.3e}")
        item(f"OR = {digital['odds_ratio_ci']:.3f}, 95% CI [{digital['ci_low']:.3f}, {digital['ci_high']:.3f}]"
             f"  (sample OR {digital['odds_ratio_sample']:.3f}; "
             f"Haldane correction {'applied' if digital['haldane_corrected'] else 'not needed'})")
        if digital_strict is not None:
            item(f"strict (polysemous anchors dropped): "
                 f"A {digital_strict['posts_with_a']}/{digital_strict['n_posts_a']} "
                 f"({digital_strict['pct_posts_a']:.1f}%) vs "
                 f"B {digital_strict['posts_with_b']}/{digital_strict['n_posts_b']} "
                 f"({digital_strict['pct_posts_b']:.1f}%), "
                 f"OR = {digital_strict['odds_ratio_ci']:.3f} "
                 f"[{digital_strict['ci_low']:.3f}, {digital_strict['ci_high']:.3f}], "
                 f"p = {digital_strict['fisher_p']:.3e}")

    # ---- fragility ---------------------------------------------------------
    banner("Fragility check - read this before quoting any G2")
    concentrated = keyness[
        (keyness["q_value"] < fdr_q) & (keyness["top_post_share_key"] >= 0.5)
    ]
    item("counted in the corpus the word is KEY TO - the only side that can inflate G2")
    item(f"significant words with >=50% of their tokens in a SINGLE post: {len(concentrated)}")
    for _, row in concentrated.sort_values("g2", ascending=False).head(10).iterrows():
        item(f"{str(row['word'])[:21]:<22}key to {row['direction']}  G2={row['g2']:>7.1f}  "
             f"single-post share={row['top_post_share_key']:.0%}  df {row['df_a']}/{row['df_b']}")
    item("G2 p-values assume token independence; posts violate it. Prefer the")
    item("post-level prevalence table for any claim about how WIDESPREAD a term is.")

    # ---- summary -----------------------------------------------------------
    summary = {
        "params": {
            "column": args.column,
            "min_freq": args.min_freq,
            "simple_maths_k": args.simple_maths_k,
            "alpha": args.alpha,
            "fdr_q": fdr_q,
            "norm_per": NORM_PER,
            "zero_adjust": ZERO_ADJUST,
        },
        "corpus_sizes": {
            label: {"posts": c.n_posts, "tokens": c.n_tokens, "vocab": len(c.freq)}
            for label, c in counts.items()
        },
        "vocabulary_keyness": {
            "n_tested": int(len(keyness)),
            "n_significant": int((keyness["q_value"] < fdr_q).sum()),
            "n_significant_a": int(((keyness["q_value"] < fdr_q) & (keyness["direction"] == "A")).sum()),
            "n_significant_b": int(((keyness["q_value"] < fdr_q) & (keyness["direction"] == "B")).sum()),
            "top_a": a_key.head(args.top)[["word", "freq_a", "freq_b", "g2", "log_ratio", "q_value"]]
                          .to_dict("records"),
            "top_b": b_key.head(args.top)[["word", "freq_a", "freq_b", "g2", "log_ratio", "q_value"]]
                          .to_dict("records"),
            "n_single_post_dominated": int(len(concentrated)),
        },
        "anchor_keyness": {
            "n_anchors": int(len(anchor_keyness)),
            "n_corpus_specific": int(len(corpus_specific)),
            "n_significant": int((anchor_keyness["q_value"].fillna(1.0) < fdr_q).sum()),
        },
        "domain_keyness": domain_keyness.to_dict("records"),
        "domain_prevalence": domain_prevalence,
        "digital_prevalence_headline": digital,
        "caveats": [
            "G2 assumes token independence; tokens within a post are dependent. "
            "q-values rank, they do not calibrate.",
            "Log Ratio with a zero cell is bounded by the 0.5 adjustment, not by data; "
            "see log_ratio_sm and the zero_cell flag.",
            "Domain totals labelled 'all' include polysemous anchors "
            "(e.g. digital: connection, disconnect, signal); 'strict' excludes them.",
        ],
    }
    save_summary(summary, out_path("summaries", "p2", 10, "keyness_prevalence", ext="json"))

    banner("Done")
    for name in ("keyness", "keyness_top100__a", "keyness_top100__b",
                 "anchor_keyness", "domain_keyness", "prevalence"):
        item(f"outputs/tables/p2_10_{name}.csv")
    item("outputs/summaries/p2_10_keyness_prevalence.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
