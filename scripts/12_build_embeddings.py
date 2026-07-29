"""Phase 2 - build comparable representations for anchors and domains.

The research question is whether migration narratives organise a concept the
same way grief narratives do. That requires a representation of each anchor (or
domain) IN EACH CORPUS that is comparable across corpora.

Two backends with deliberately orthogonal failure modes, so agreement between
them is evidence rather than a tautology:

  ppmi (default)
      A PPMI context profile over a SHARED basis - the vocabulary items frequent
      enough in BOTH corpora. Comparison is second-order (cosine between two
      profiles over the same basis), so it is alignment-free by construction:
      no Procrustes rotation is estimated and none is needed.

  contextual
      Token embeddings from a frozen pretrained transformer, averaged into a
      per-corpus sense vector (Giulianelli et al. 2020; Martinc et al. 2020).
      Both corpora are encoded by the SAME frozen model, so they share one space
      by construction and again no alignment step exists.

Why not Word2Vec/FastText, which the author's brief mentions: measured
split-half neighbour agreement on these corpora sits at the chance baseline
below f=200. SGNS estimates hundreds of free parameters per word from those same
counts, so at this corpus size it returns noise. A frozen encoder MEASURES
already-trained vectors instead of ESTIMATING new ones, which is why it degrades
gracefully where SGNS does not.

Why not Procrustes alignment: it must be estimated from shared words that are
individually stable, and on this data they are not. Fitting a rotation from
unstable points manufactures apparent shift - the error Dubossarsky et al.
(2017) showed underlies much published "semantic change".

The DOMAIN is the primary unit: only 7 of 71 anchors clear the Tier-1 floor,
but 4 of 4 domains do when their anchors are pooled.

Example:
    ...\\python.exe scripts/12_build_embeddings.py
    ...\\python.exe scripts/12_build_embeddings.py --backend contextual --unit anchor
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
from analysis_io import as_token_list, banner, item, save_summary  # noqa: E402
from contextual_vectors import (  # noqa: E402
    all_but_the_top,
    average_profiles,
    encode_corpus_token_cache,
    load_token_cache,
    type_embeddings,
)
from vectors import ppmi  # noqa: E402

CORPORA = ("a", "b")


# --------------------------------------------------------------------------
# shared basis
# --------------------------------------------------------------------------


def load_documents(analysis_dir: Path, label: str, column: str = "tokens_lemma") -> list[list[str]]:
    df = pd.read_parquet(analysis_dir / f"corpus_{label}.parquet", columns=[column])
    return [as_token_list(tokens) for tokens in df[column]]


def shared_basis(
    documents: dict[str, list[list[str]]], *, min_freq: int
) -> tuple[str, ...]:
    """Vocabulary items frequent enough in BOTH corpora.

    Using a corpus-specific basis would make the two profiles incommensurable:
    they would be vectors over different coordinate systems.
    """
    from collections import Counter

    counters = {}
    for label, docs in documents.items():
        counter: Counter = Counter()
        for tokens in docs:
            counter.update(tokens)
        counters[label] = counter

    shared = {
        word for word in counters["a"]
        if counters["a"][word] >= min_freq and counters["b"].get(word, 0) >= min_freq
    }
    return tuple(sorted(shared))


# --------------------------------------------------------------------------
# PPMI context profiles
# --------------------------------------------------------------------------


def document_count_matrix(
    documents: list[list[str]],
    targets: dict[str, tuple[str, ...]],
    basis: tuple[str, ...],
    *,
    window: int,
):
    """Per-document co-occurrence contributions as a sparse (n_docs, T*B) matrix.

    Counting is additive over documents, so any subset's profile is just the sum
    of its rows. That turns resampling - split-half reliability, bootstrap CIs,
    and especially a 1,000-iteration permutation test - from "rebuild the whole
    profile each time" into a row-sum, which is the difference between an hour
    and a few seconds.

    Row-major layout within a row: target_row * len(basis) + basis_column.
    """
    import scipy.sparse as sp

    basis_index = {word: i for i, word in enumerate(basis)}
    names = sorted(targets)
    row_of = {name: i for i, name in enumerate(names)}
    n_basis = len(basis)

    rows_for_term: dict[str, list[int]] = {}
    for name, terms in targets.items():
        for term in terms:
            rows_for_term.setdefault(term, []).append(row_of[name])

    indptr = [0]
    indices: list[int] = []
    data: list[float] = []

    for tokens in documents:
        local: dict[int, float] = {}
        n = len(tokens)
        for position, token in enumerate(tokens):
            hit = rows_for_term.get(token)
            if not hit:
                continue
            low = max(0, position - window)
            high = min(n, position + window + 1)
            for other in range(low, high):
                if other == position:
                    continue
                column = basis_index.get(tokens[other])
                if column is None:
                    continue
                for target_row in hit:
                    key = target_row * n_basis + column
                    local[key] = local.get(key, 0.0) + 1.0
        for key in sorted(local):
            indices.append(key)
            data.append(local[key])
        indptr.append(len(indices))

    matrix = sp.csr_matrix(
        (data, indices, indptr), shape=(len(documents), len(names) * n_basis), dtype=np.float64
    )
    return matrix, names


def profiles_from_counts(matrix, names: list[str], n_basis: int, *, shift_k: int, cds_smoothing: float):
    """Sum selected document rows into PPMI profiles."""
    import scipy.sparse as sp

    totals = np.asarray(matrix.sum(axis=0)).ravel().reshape(len(names), n_basis)
    weighted = ppmi(sp.csr_matrix(totals), shift_k=shift_k, cds_smoothing=cds_smoothing).toarray()
    return {name: weighted[i] for i, name in enumerate(names)}


def ppmi_profiles(
    documents: list[list[str]],
    targets: dict[str, tuple[str, ...]],
    basis: tuple[str, ...],
    *,
    window: int,
    shift_k: int,
    cds_smoothing: float,
) -> dict[str, np.ndarray]:
    """PPMI profile over `basis` for each target.

    A target maps to one or more surface terms; a domain pools all of its
    anchors, which is what lifts a domain above the frequency floor when its
    individual anchors sit below it.
    """
    basis_index = {word: i for i, word in enumerate(basis)}

    # rows = targets, cols = basis. Counted jointly so one pass builds all.
    names = sorted(targets)
    row_of = {name: i for i, name in enumerate(names)}
    counts = np.zeros((len(names), len(basis)), dtype=np.float64)

    # term -> row indices. Precomputed so the token loop is O(1) per token
    # rather than O(n_targets); at 71 anchors over 232k tokens the scan form
    # costs ~16M set lookups per profile build, which the reliability gate
    # then repeats for every split.
    rows_for_term: dict[str, list[int]] = {}
    for name, terms in targets.items():
        for term in terms:
            rows_for_term.setdefault(term, []).append(row_of[name])

    for tokens in documents:
        n = len(tokens)
        for position, token in enumerate(tokens):
            hit = rows_for_term.get(token)
            if not hit:
                continue
            low = max(0, position - window)
            high = min(n, position + window + 1)
            for other in range(low, high):
                if other == position:
                    continue
                column = basis_index.get(tokens[other])
                if column is None:
                    continue
                for row in hit:
                    counts[row, column] += 1.0

    import scipy.sparse as sp

    weighted = ppmi(
        sp.csr_matrix(counts), shift_k=shift_k, cds_smoothing=cds_smoothing
    ).toarray()
    return {name: weighted[row_of[name]] for name in names}


# --------------------------------------------------------------------------


def build_targets(anchors, unit: str) -> dict[str, tuple[str, ...]]:
    if unit == "domain":
        return {name: domain.terms for name, domain in anchors.domains.items()}
    if unit == "anchor":
        return {anchor.term: (anchor.term,) for anchor in anchors.anchors}
    raise ValueError(f"unknown unit {unit!r}")


def _all_anchor_terms(anchors) -> set[str]:
    return {anchor.term for anchor in anchors.anchors}


def main() -> int:
    parser = argparse.ArgumentParser(description="Build comparable anchor/domain representations")
    parser.add_argument("--backend", choices=["ppmi", "contextual"], default=None)
    parser.add_argument("--unit", choices=["domain", "anchor", "both"], default=None)
    parser.add_argument("--analysis-dir", type=Path, default=None)
    parser.add_argument("--min-basis-freq", type=int, default=None)
    parser.add_argument("--window", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = load_analysis_config(seed=args.seed)
    anchors = load_anchors()
    analysis_dir = args.analysis_dir or cfg.path("analysis_dir")
    backend = args.backend or cfg.get("vectors.backend")
    units = ["domain", "anchor"] if (args.unit or cfg.get("vectors.unit")) == "both" else [args.unit or cfg.get("vectors.unit")]

    window = args.window or int(cfg.get("vectors.ppmi.window"))
    min_basis = args.min_basis_freq or int(cfg.get("vectors.ppmi.shared_basis_min_freq"))
    shift_k = int(cfg.get("vectors.ppmi.shift_k"))
    smoothing = float(cfg.get("vectors.ppmi.cds_smoothing"))

    vectors_dir = analysis_dir / "vectors"
    vectors_dir.mkdir(parents=True, exist_ok=True)

    documents = {label: load_documents(analysis_dir, label) for label in CORPORA}
    basis = shared_basis(documents, min_freq=min_basis)

    banner("Shared basis")
    item(f"min frequency in BOTH corpora: {min_basis}")
    item(f"basis dimensions: {len(basis)}")
    if len(basis) < 50:
        item("WARNING: basis is very small; profiles will be poorly estimated")

    abtt_k = int(cfg.get("vectors.contextual.all_but_the_top"))
    summary: dict = {
        "backend": backend,
        "basis_size": len(basis),
        "basis_min_freq": min_basis,
        "units": {},
    }

    # Contextual: encode once per corpus into a token cache, then average.
    caches: dict[str, tuple] = {}
    if backend == "contextual":
        needed_terms = _all_anchor_terms(anchors) | set(basis)
        banner("Contextual token cache")
        item(f"model: {cfg.get('vectors.contextual.model_name')}")
        item(f"needed_terms: {len(needed_terms)} (anchors + shared basis)")
        for label in CORPORA:
            encode_corpus_token_cache(
                analysis_dir,
                label,
                needed_terms,
                model_name=str(cfg.get("vectors.contextual.model_name")),
                layers=tuple(int(x) for x in cfg.get("vectors.contextual.layers")),
                batch_size=int(cfg.get("vectors.contextual.batch_size")),
                max_seq_length=int(cfg.get("vectors.contextual.max_seq_length")),
                force=False,
            )
            caches[label] = load_token_cache(analysis_dir, label)

        # Shared-basis type embeddings for neighbour lists (both units reuse).
        banner("Type embeddings (shared basis)")
        for label in CORPORA:
            doc_idx, token_str, vectors = caches[label]
            type_names, type_matrix = type_embeddings(
                doc_idx, token_str, vectors, basis
            )
            if abtt_k > 0 and len(type_names) > abtt_k:
                type_dict = {
                    name: type_matrix[i] for i, name in enumerate(type_names)
                }
                type_dict = all_but_the_top(type_dict, abtt_k)
                type_names = tuple(sorted(type_dict))
                type_matrix = np.vstack([type_dict[n] for n in type_names])
            destination = vectors_dir / f"type_{label}_contextual.npz"
            np.savez_compressed(
                destination,
                matrix=type_matrix.astype(np.float32),
                names=np.array(type_names, dtype=object),
                basis=np.array(basis, dtype=object),
            )
            item(
                f"wrote {destination.relative_to(ROOT)}  "
                f"types={len(type_names)}/{len(basis)}"
            )
        summary["type_embeddings"] = {
            label: int(np.load(vectors_dir / f"type_{label}_contextual.npz", allow_pickle=True)["names"].shape[0])
            for label in CORPORA
        }
        summary["contextual"] = {
            "model_name": str(cfg.get("vectors.contextual.model_name")),
            "layers": list(cfg.get("vectors.contextual.layers")),
            "all_but_the_top": abtt_k,
            "needed_terms": len(needed_terms),
        }

    for unit in units:
        targets = build_targets(anchors, unit)
        banner(f"Build {backend} profiles - unit={unit} ({len(targets)} targets)")

        profiles: dict[str, dict[str, np.ndarray]] = {}
        for label in CORPORA:
            if backend == "ppmi":
                profiles[label] = ppmi_profiles(
                    documents[label], targets, basis,
                    window=window, shift_k=shift_k, cds_smoothing=smoothing,
                )
            else:
                doc_idx, token_str, vectors = caches[label]
                profiles[label] = average_profiles(
                    doc_idx, token_str, vectors, targets
                )
                # ABTT needs more points than PCs; domain unit has only 4 targets.
                if abtt_k > 0 and len(profiles[label]) > abtt_k + 2:
                    profiles[label] = all_but_the_top(profiles[label], abtt_k)
            item(f"corpus {label}: {len(profiles[label])} profiles")

        names = sorted(targets)
        for label in CORPORA:
            matrix = np.vstack([profiles[label][n] for n in names])
            destination = vectors_dir / f"{unit}_{label}_{backend}.npz"
            np.savez_compressed(
                destination,
                matrix=matrix,
                names=np.array(names, dtype=object),
                basis=np.array(basis, dtype=object) if backend == "ppmi" else np.array([], dtype=object),
            )
            nonzero = int((np.linalg.norm(matrix, axis=1) > 0).sum())
            item(f"wrote {destination.relative_to(ROOT)}  shape={matrix.shape}  non-empty={nonzero}/{len(names)}")

        summary["units"][unit] = {
            "n_targets": len(targets),
            "targets": names,
            "empty_profiles": {
                label: [n for n in names if np.linalg.norm(profiles[label][n]) == 0]
                for label in CORPORA
            },
        }

    save_summary(summary, analysis_dir / f"vectors/build_summary_{backend}.json")
    banner("Done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
