"""Phases 2-4 - co-occurrence, PPMI, similarity, neighbour ranking, reliability.

Shared module for scripts 07-21. Pure functions, no I/O, no global state.

These functions produce the numbers that go in the paper, so two properties are
enforced throughout:

* determinism - `top_k_neighbors` breaks ties alphabetically, because an
  unstable tie-break makes the neighbour table change between identical runs;
* honesty - `jaccard_at_k` is always reported next to `jaccard_chance`, since a
  Jaccard of 0.02 on a 428-type vocabulary IS chance and looks like a finding
  without it.

No gensim: Word2Vec is indefensible at 28k/102k tokens (measured split-half
neighbour agreement sits at chance below f=200), and gensim pins numpy<2, which
would break the spaCy binary stack. Count-based PPMI degrades far more
gracefully and is exactly reproducible.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
import scipy.sparse as sp

EPSILON = 1e-12


# --------------------------------------------------------------------------
# co-occurrence
# --------------------------------------------------------------------------


def build_vocabulary(
    documents: Iterable[Sequence[str]],
    *,
    min_count: int = 1,
    max_size: int | None = None,
) -> tuple[str, ...]:
    """Sorted vocabulary above `min_count`. Sorted, so column order is reproducible."""
    counts: Counter = Counter()
    for tokens in documents:
        counts.update(str(t) for t in tokens)
    kept = [w for w, c in counts.items() if c >= min_count]
    kept.sort(key=lambda w: (-counts[w], w))
    if max_size is not None:
        kept = kept[:max_size]
    return tuple(sorted(kept))


def build_cooccurrence(
    documents: Iterable[Sequence[str]],
    vocabulary: Sequence[str],
    *,
    window: int = 5,
    symmetric: bool = True,
) -> sp.csr_matrix:
    """Sparse |V|x|V| co-occurrence counts within +/-`window` tokens.

    Counting never crosses a document boundary - each document is scanned
    independently, so the last token of one post cannot co-occur with the first
    token of the next.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    index = {word: i for i, word in enumerate(vocabulary)}
    size = len(vocabulary)

    rows: list[int] = []
    cols: list[int] = []

    for tokens in documents:
        ids = [index.get(str(t), -1) for t in tokens]
        n = len(ids)
        for position, centre in enumerate(ids):
            if centre < 0:
                continue
            hi = min(n, position + window + 1)
            for other in range(position + 1, hi):
                context = ids[other]
                if context < 0:
                    continue
                rows.append(centre)
                cols.append(context)
                if symmetric:
                    rows.append(context)
                    cols.append(centre)

    data = np.ones(len(rows), dtype=np.float64)
    matrix = sp.coo_matrix((data, (rows, cols)), shape=(size, size)).tocsr()
    matrix.sum_duplicates()
    return matrix


def ppmi(
    counts: sp.spmatrix,
    *,
    shift_k: int = 1,
    cds_smoothing: float = 1.0,
    positive: bool = True,
) -> sp.csr_matrix:
    """Positive Pointwise Mutual Information.

        pmi(i,j) = log[ p(i,j) / (p(i) * p_ctx(j)) ] - log(shift_k)

    `cds_smoothing` is the context-distribution smoothing exponent of Levy,
    Goldberg & Dagan (2015); 1.0 is plain PPMI, 0.75 flattens the context
    marginal. Measured on this project's data, 1.0 outperformed 0.75.
    """
    matrix = sp.csr_matrix(counts, dtype=np.float64)
    total = matrix.sum()
    if total <= 0:
        return sp.csr_matrix(matrix.shape, dtype=np.float64)

    row_sums = np.asarray(matrix.sum(axis=1)).ravel()
    col_sums = np.asarray(matrix.sum(axis=0)).ravel()

    if cds_smoothing != 1.0:
        smoothed = np.power(col_sums, cds_smoothing)
        col_prob = smoothed / max(smoothed.sum(), EPSILON)
    else:
        col_prob = col_sums / total

    row_prob = row_sums / total

    coo = matrix.tocoo()
    joint = coo.data / total
    denominator = row_prob[coo.row] * col_prob[coo.col]

    with np.errstate(divide="ignore", invalid="ignore"):
        values = np.log(np.where(denominator > 0, joint / np.maximum(denominator, EPSILON), 1.0))
    if shift_k > 1:
        values = values - math.log(shift_k)
    if positive:
        values = np.maximum(values, 0.0)

    keep = values > 0
    result = sp.coo_matrix(
        (values[keep], (coo.row[keep], coo.col[keep])), shape=matrix.shape
    ).tocsr()
    result.eliminate_zeros()
    return result


def svd_embed(matrix: sp.spmatrix, *, dims: int = 200, seed: int = 0) -> np.ndarray:
    """Truncated SVD of a (PPMI) matrix. Optional - raw PPMI measured better here."""
    from sklearn.decomposition import TruncatedSVD

    n_components = min(dims, min(matrix.shape) - 1)
    if n_components < 1:
        raise ValueError(f"cannot take {dims} components from shape {matrix.shape}")
    model = TruncatedSVD(n_components=n_components, random_state=seed)
    return model.fit_transform(matrix)


# --------------------------------------------------------------------------
# similarity
# --------------------------------------------------------------------------


def l2_normalize(matrix: np.ndarray, *, axis: int = 1) -> np.ndarray:
    """Unit-norm rows. Zero rows stay zero rather than becoming NaN."""
    array = np.asarray(matrix, dtype=np.float64)
    norms = np.linalg.norm(array, axis=axis, keepdims=True)
    return array / np.where(norms > 0, norms, 1.0)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two vectors. Returns 0.0 if either is the zero vector."""
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    if x.shape != y.shape:
        raise ValueError(f"shape mismatch: {x.shape} vs {y.shape}")
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denominator <= 0:
        return 0.0
    return float(np.dot(x, y) / denominator)


def cosine_matrix(matrix: np.ndarray, other: np.ndarray | None = None) -> np.ndarray:
    """Pairwise cosine similarities between rows."""
    left = l2_normalize(np.asarray(matrix, dtype=np.float64))
    right = left if other is None else l2_normalize(np.asarray(other, dtype=np.float64))
    return left @ right.T


def top_k_neighbors(
    similarities: Sequence[float],
    vocabulary: Sequence[str],
    *,
    k: int = 15,
    exclude: Iterable[str] = (),
    min_similarity: float | None = None,
) -> list[tuple[str, float]]:
    """Top-k most similar vocabulary items, highest first.

    Ties break alphabetically so two identical runs return identical tables.
    """
    if len(similarities) != len(vocabulary):
        raise ValueError(
            f"similarities ({len(similarities)}) and vocabulary ({len(vocabulary)}) differ in length"
        )
    blocked = {str(e).lower() for e in exclude}
    scored = [
        (str(word), float(score))
        for word, score in zip(vocabulary, similarities)
        if str(word).lower() not in blocked
        and not (isinstance(score, float) and math.isnan(score))
        and (min_similarity is None or score >= min_similarity)
    ]
    scored.sort(key=lambda pair: (-pair[1], pair[0]))
    return scored[:k]


# --------------------------------------------------------------------------
# neighbourhood comparison
# --------------------------------------------------------------------------


def jaccard_at_k(left: Sequence[str], right: Sequence[str], *, k: int | None = None) -> float:
    """Jaccard overlap of two neighbour lists. Always report with `jaccard_chance`."""
    a = set(left[:k] if k else left)
    b = set(right[:k] if k else right)
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def jaccard_chance(k: int, vocabulary_size: int) -> float:
    """Expected Jaccard for two random k-subsets of a shared vocabulary: k/(2V-k).

    On this project's 428-type shared vocabulary at k=15 this is 0.018 - and
    measured corpus-trained agreement below f=200 sits at exactly that value.
    """
    if vocabulary_size <= 0 or k <= 0:
        return 0.0
    denominator = 2 * vocabulary_size - k
    return k / denominator if denominator > 0 else 0.0


def rank_biased_overlap(
    left: Sequence[str],
    right: Sequence[str],
    *,
    p: float = 0.9,
    k: int | None = None,
    normalize: bool = True,
) -> float:
    """Rank-biased overlap (Webber, Moffat & Zobel 2010), truncated at depth k.

    Top-weighted and defined for non-conjoint lists, so it is strictly more
    informative than Jaccard for ranked neighbour lists.

    `normalize=True` (default) rescales by the maximum achievable at this depth,
    `1 - p**depth`, so identical lists score exactly 1.0. The unnormalised
    truncated form (RBO_min) caps well below 1 - at depth 15 with p=0.9 two
    identical lists score only 0.79 - which reads as disagreement in a results
    table when it is nothing of the sort. Pass normalize=False for the raw
    lower bound as originally defined.
    """
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must be in (0,1), got {p}")
    depth = k or max(len(left), len(right))
    if depth == 0:
        return 0.0

    seen_left: set[str] = set()
    seen_right: set[str] = set()
    total = 0.0
    for d in range(1, depth + 1):
        if d <= len(left):
            seen_left.add(str(left[d - 1]))
        if d <= len(right):
            seen_right.add(str(right[d - 1]))
        agreement = len(seen_left & seen_right) / d
        total += (p ** (d - 1)) * agreement

    raw = (1.0 - p) * total
    if not normalize:
        return raw
    ceiling = 1.0 - p**depth
    return raw / ceiling if ceiling > EPSILON else raw


# --------------------------------------------------------------------------
# reliability
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Reliability:
    """Split-half reliability for one target, with the Spearman-Brown estimate."""

    target: str
    n_splits: int
    mean_split_half: float
    spearman_brown: float
    std: float

    @property
    def usable(self) -> bool:
        return self.spearman_brown >= 0.25


def spearman_brown(split_half: float) -> float:
    """Project a split-half correlation to full-length reliability: 2r/(1+r)."""
    if split_half <= -1.0:
        return 0.0
    value = (2.0 * split_half) / (1.0 + split_half)
    return float(min(max(value, 0.0), 1.0))


def split_half_reliability(
    build_vector,
    documents: Sequence[Sequence[str]],
    target: str,
    *,
    n_splits: int = 50,
    rng: np.random.Generator | None = None,
) -> Reliability:
    """Estimate how reliably `target`'s vector can be measured from this corpus.

    `build_vector(subset) -> np.ndarray | None` is called on each random half.
    Splits are over DOCUMENTS, never tokens: tokens within a post share an
    author, an event and lexical priming, so token-level resampling would
    understate variance and inflate significance.

    This is the gate that determines whether neighbour lists are reportable at
    all. On this project's corpora, measured agreement below f=200 is at chance.
    """
    generator = rng or np.random.default_rng(0)
    n = len(documents)
    if n < 4:
        return Reliability(target, 0, 0.0, 0.0, 0.0)

    scores: list[float] = []
    for _ in range(n_splits):
        order = generator.permutation(n)
        midpoint = n // 2
        first = [documents[i] for i in order[:midpoint]]
        second = [documents[i] for i in order[midpoint:]]
        left = build_vector(first)
        right = build_vector(second)
        if left is None or right is None:
            continue
        scores.append(cosine(left, right))

    if not scores:
        return Reliability(target, 0, 0.0, 0.0, 0.0)

    mean = float(np.mean(scores))
    return Reliability(
        target=target,
        n_splits=len(scores),
        mean_split_half=mean,
        spearman_brown=spearman_brown(mean),
        std=float(np.std(scores)),
    )


def attenuation_correct(observed: float, reliability_a: float, reliability_b: float) -> float:
    """Correct a cross-corpus cosine for measurement unreliability (Spearman 1904).

        cos* = cos(v_A, v_B) / sqrt(rel_A * rel_B)

    A raw cosine is uninterpretable when a vector barely correlates with itself
    across split-halves. cos* expresses the observed value as a fraction of the
    highest value the data could possibly produce, and is capped at 1.0.

    Unstable when either reliability is small - which is exactly why Tier-3
    anchors are excluded rather than corrected.
    """
    ceiling = math.sqrt(max(reliability_a, 0.0) * max(reliability_b, 0.0))
    if ceiling <= EPSILON:
        return float("nan")
    return float(min(observed / ceiling, 1.0))


def reliability_ceiling(reliability_a: float, reliability_b: float) -> float:
    """The highest cosine this pair could show: sqrt(rel_A * rel_B)."""
    return math.sqrt(max(reliability_a, 0.0) * max(reliability_b, 0.0))


# --------------------------------------------------------------------------
# topic coherence
# --------------------------------------------------------------------------


def umass_coherence(
    topic_terms: Sequence[str],
    document_frequency: Mapping[str, int],
    codocument_frequency: Mapping[tuple[str, str], int],
    *,
    epsilon: float = 1.0,
) -> float:
    """UMass coherence for one topic.

        C = sum_{i<j} log[ (D(wi,wj) + eps) / D(wj) ]

    Computed from document-level counts the co-occurrence step already produces,
    so no gensim dependency is needed.
    """
    terms = [str(t) for t in topic_terms]
    if len(terms) < 2:
        return 0.0
    total = 0.0
    pairs = 0
    for i in range(1, len(terms)):
        for j in range(i):
            wi, wj = terms[i], terms[j]
            key = (wi, wj) if (wi, wj) in codocument_frequency else (wj, wi)
            co = codocument_frequency.get(key, 0)
            denominator = document_frequency.get(wj, 0)
            if denominator <= 0:
                continue
            total += math.log((co + epsilon) / denominator)
            pairs += 1
    return total / pairs if pairs else 0.0


def codocument_counts(
    documents: Iterable[Sequence[str]],
    vocabulary: Iterable[str],
) -> tuple[dict[str, int], dict[tuple[str, str], int]]:
    """Document frequency and co-document frequency for a term set."""
    wanted = {str(w) for w in vocabulary}
    df: Counter = Counter()
    codf: Counter = Counter()
    for tokens in documents:
        present = sorted(wanted & {str(t) for t in tokens})
        df.update(present)
        for i in range(len(present)):
            for j in range(i + 1, len(present)):
                codf[(present[i], present[j])] += 1
    return dict(df), dict(codf)
