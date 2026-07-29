"""Word2Vec / FastText helpers for empirical small-corpus evaluation.

These models are trained FROM the research corpora (unlike frozen BERT).
Spaces from independent trainings are not aligned, so cross-half and
cross-corpus comparison uses SECOND-ORDER profiles: cosine of the target
vector to every type in a shared basis. That is alignment-free and
comparable to the PPMI design.

Neighbour lists use gensim similarity within a single trained model
(restricted to the shared basis when possible).

Gensim 4.4+ supports numpy 2.x; pin is no longer a blocker for this stack.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np

from vectors import cosine, top_k_neighbors

EPSILON = 1e-12


def train_keyed_vectors(
    documents: Sequence[Sequence[str]],
    *,
    algorithm: str,
    vector_size: int = 100,
    window: int = 5,
    min_count: int = 5,
    epochs: int = 20,
    sg: int = 1,
    negative: int = 5,
    seed: int = 0,
    workers: int = 1,
):
    """Train Word2Vec or FastText; return gensim KeyedVectors (wv)."""
    algorithm = algorithm.lower().replace("-", "")
    # Materialise token lists; gensim iterates multiple epochs.
    sentences = [[str(t) for t in doc] for doc in documents if doc]
    if not sentences:
        raise ValueError("no documents to train on")

    common = dict(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        epochs=epochs,
        sg=sg,
        negative=negative,
        seed=seed,
        workers=workers,
    )
    if algorithm in ("word2vec", "w2v", "sgns"):
        from gensim.models import Word2Vec

        model = Word2Vec(**common)
        return model.wv
    if algorithm in ("fasttext", "ft"):
        from gensim.models import FastText

        model = FastText(**common)
        return model.wv
    raise ValueError(f"unknown algorithm {algorithm!r}; use word2vec or fasttext")


def target_vector(
    kv,
    terms: Sequence[str],
) -> np.ndarray | None:
    """Mean embedding of terms present in the model. None if none present."""
    vectors = []
    for term in terms:
        token = str(term)
        if token in kv:
            vectors.append(np.asarray(kv[token], dtype=np.float64))
        else:
            # FastText can synthesise OOV via subwords when available.
            try:
                vectors.append(np.asarray(kv.get_vector(token), dtype=np.float64))
            except KeyError:
                continue
    if not vectors:
        return None
    return np.mean(vectors, axis=0)


def second_order_profile(
    kv,
    terms: Sequence[str],
    basis: Sequence[str],
) -> np.ndarray:
    """Alignment-free profile: cos(target, basis_j) for each shared basis type.

    Missing basis types contribute 0. Missing target yields the zero vector.
    """
    profile = np.zeros(len(basis), dtype=np.float64)
    centre = target_vector(kv, terms)
    if centre is None or float(np.linalg.norm(centre)) <= EPSILON:
        return profile
    for i, word in enumerate(basis):
        token = str(word)
        if token not in kv:
            continue
        other = np.asarray(kv[token], dtype=np.float64)
        profile[i] = cosine(centre, other)
    return profile


def neighbors_from_kv(
    kv,
    terms: Sequence[str],
    *,
    k: int = 15,
    exclude: Iterable[str] = (),
    restrict_to: Sequence[str] | None = None,
) -> list[tuple[str, float]]:
    """Top-k nearest types to the target mean vector.

    If `restrict_to` is set (shared basis), only those types are ranked so
    neighbour pools are comparable across models and to PPMI/BERT.
    """
    centre = target_vector(kv, terms)
    if centre is None:
        return []
    exclude_set = {str(x) for x in exclude} | {str(t) for t in terms}

    if restrict_to is not None:
        scores: list[float] = []
        vocab: list[str] = []
        for word in restrict_to:
            token = str(word)
            if token in exclude_set or token not in kv:
                continue
            scores.append(cosine(centre, np.asarray(kv[token], dtype=np.float64)))
            vocab.append(token)
        return top_k_neighbors(scores, vocab, k=k, exclude=())

    # Fall back to gensim's full-vocab most_similar when no restriction.
    try:
        raw = kv.most_similar(positive=[centre], topn=k + len(exclude_set) + 5)
    except Exception:
        return []
    out: list[tuple[str, float]] = []
    for word, score in raw:
        if word in exclude_set:
            continue
        out.append((str(word), float(score)))
        if len(out) >= k:
            break
    return out


def profiles_for_targets(
    kv,
    targets: Mapping[str, tuple[str, ...]],
    basis: Sequence[str],
) -> dict[str, np.ndarray]:
    return {
        name: second_order_profile(kv, terms, basis)
        for name, terms in targets.items()
    }
