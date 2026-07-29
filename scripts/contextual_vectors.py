"""Phase 2 - document-level BERT token vectors with disk cache.

Efficient contextual path used by scripts 12-14:

1. Encode each document once with a frozen AutoModel (mean of configured layers).
2. Mean-pool wordpieces that map back to each whitespace token (via word_ids).
3. Cache only tokens that appear in a `needed_terms` set (anchors + shared basis).
4. Downstream steps re-average profiles under document masks - split-half,
   bootstrap, stratified permutation - WITHOUT re-encoding.

No argparse main; pure helpers + cache I/O.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analysis_io import as_token_list, item  # noqa: E402
from vectors import l2_normalize, top_k_neighbors  # noqa: E402

CACHE_SUBDIR = "vectors/contextual_cache"


# --------------------------------------------------------------------------
# paths / cache I/O
# --------------------------------------------------------------------------


def cache_dir(analysis_dir: Path) -> Path:
    path = analysis_dir / "vectors" / "contextual_cache"
    path.mkdir(parents=True, exist_ok=True)
    return path


def cache_paths(analysis_dir: Path, label: str) -> tuple[Path, Path]:
    base = cache_dir(analysis_dir)
    return base / f"meta_{label}.json", base / f"token_vectors_{label}.npz"


def load_token_cache(
    analysis_dir: Path, label: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load (doc_idx int32, token_str object, vectors float32)."""
    meta_path, vec_path = cache_paths(analysis_dir, label)
    if not vec_path.exists():
        raise FileNotFoundError(
            f"missing contextual token cache for corpus {label}: {vec_path}\n"
            "Run: python scripts/12_build_embeddings.py --backend contextual --unit both"
        )
    with np.load(vec_path, allow_pickle=True) as archive:
        doc_idx = np.asarray(archive["doc_idx"], dtype=np.int32)
        token_str = np.asarray(archive["token_str"], dtype=object)
        vectors = np.asarray(archive["vectors"], dtype=np.float32)
    return doc_idx, token_str, vectors


def load_meta(analysis_dir: Path, label: str) -> dict:
    meta_path, _ = cache_paths(analysis_dir, label)
    if not meta_path.exists():
        raise FileNotFoundError(f"missing contextual cache meta: {meta_path}")
    return json.loads(meta_path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# document-level encoding
# --------------------------------------------------------------------------


def encode_corpus_token_cache(
    analysis_dir: Path,
    label: str,
    needed_terms: set[str] | Iterable[str],
    *,
    model_name: str = "bert-base-uncased",
    layers: Sequence[int] = (-1, -2, -3, -4),
    batch_size: int = 16,
    max_seq_length: int = 256,
    force: bool = False,
) -> Path:
    """Encode `corpus_{label}.parquet` tokens_ctx; write token-vector cache.

    Only whitespace tokens present in `needed_terms` are stored. Returns the
    path of the written `.npz` file.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    needed = {str(t) for t in needed_terms}
    meta_path, vec_path = cache_paths(analysis_dir, label)

    if not force and vec_path.exists() and meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if (
                meta.get("model_name") == model_name
                and list(meta.get("layers", [])) == list(layers)
                and int(meta.get("max_seq_length", -1)) == int(max_seq_length)
            ):
                item(f"cache hit corpus {label}: {vec_path.name} (n={meta.get('n_token_vectors')})")
                return vec_path
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass

    df = pd.read_parquet(analysis_dir / f"corpus_{label}.parquet", columns=["tokens_ctx"])
    documents = [as_token_list(tokens) for tokens in df["tokens_ctx"]]
    n_docs = len(documents)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    layer_ids = tuple(int(x) for x in layers)
    doc_idx_parts: list[np.ndarray] = []
    token_str_parts: list[np.ndarray] = []
    vector_parts: list[np.ndarray] = []
    dim: int | None = None

    for start in range(0, n_docs, batch_size):
        batch_docs = documents[start : start + batch_size]
        # Skip empty docs - tokenizer rejects empty word lists.
        keep_local = [i for i, toks in enumerate(batch_docs) if toks]
        if not keep_local:
            continue
        windows = [batch_docs[i] for i in keep_local]
        global_ids = [start + i for i in keep_local]

        # Keep BatchEncoding on CPU so word_ids() stays available; move tensors.
        encoded = tokenizer(
            windows,
            is_split_into_words=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_length,
        )
        model_inputs = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            output = model(**model_inputs)
        # Mean of selected hidden layers (negative indices OK on the tuple).
        hidden = torch.stack([output.hidden_states[i] for i in layer_ids]).mean(0)
        hidden = hidden.cpu().numpy().astype(np.float32)
        if dim is None:
            dim = int(hidden.shape[-1])

        for row, doc_i in enumerate(global_ids):
            tokens = documents[doc_i]
            word_ids = encoded.word_ids(row)
            # Map word position -> list of subword indices (skip specials: None).
            pieces_for_word: dict[int, list[int]] = {}
            for piece_i, word_i in enumerate(word_ids):
                if word_i is None:
                    continue
                pieces_for_word.setdefault(word_i, []).append(piece_i)

            row_doc: list[int] = []
            row_tok: list[str] = []
            row_vec: list[np.ndarray] = []
            for word_i, pieces in pieces_for_word.items():
                if word_i < 0 or word_i >= len(tokens):
                    continue
                surface = tokens[word_i]
                if surface not in needed:
                    continue
                vec = hidden[row, pieces].mean(axis=0)
                row_doc.append(doc_i)
                row_tok.append(surface)
                row_vec.append(vec)

            if row_doc:
                doc_idx_parts.append(np.asarray(row_doc, dtype=np.int32))
                token_str_parts.append(np.asarray(row_tok, dtype=object))
                vector_parts.append(np.stack(row_vec).astype(np.float32))

        if (start // batch_size) % 20 == 0:
            n_so_far = sum(len(x) for x in doc_idx_parts)
            item(
                f"corpus {label}: encoded docs {min(start + batch_size, n_docs)}/{n_docs}"
                f"  tokens_kept={n_so_far}"
            )

    if dim is None:
        # Empty corpus or model not run - still write a valid empty cache.
        # Infer dim from a one-shot empty-safe probe.
        dim = int(model.config.hidden_size)

    if doc_idx_parts:
        doc_idx = np.concatenate(doc_idx_parts)
        token_str = np.concatenate(token_str_parts)
        vectors = np.vstack(vector_parts).astype(np.float32)
    else:
        doc_idx = np.zeros(0, dtype=np.int32)
        token_str = np.zeros(0, dtype=object)
        vectors = np.zeros((0, dim), dtype=np.float32)

    np.savez_compressed(
        vec_path,
        doc_idx=doc_idx,
        token_str=token_str,
        vectors=vectors,
    )
    meta = {
        "model_name": model_name,
        "layers": list(layer_ids),
        "max_seq_length": int(max_seq_length),
        "batch_size": int(batch_size),
        "n_docs": n_docs,
        "dim": int(dim),
        "n_token_vectors": int(len(doc_idx)),
        "n_needed_terms": len(needed),
        "label": label,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    item(f"wrote cache {vec_path.relative_to(ROOT)}  tokens={len(doc_idx)}  dim={dim}")
    return vec_path


# --------------------------------------------------------------------------
# profile / type aggregation from cache
# --------------------------------------------------------------------------


def _row_mask(
    doc_idx: np.ndarray,
    doc_mask: np.ndarray | None,
) -> np.ndarray:
    if doc_mask is None:
        return np.ones(len(doc_idx), dtype=bool)
    # doc_mask indexed by document id; select cache rows whose doc is kept.
    return doc_mask[doc_idx]


def average_profiles(
    doc_idx: np.ndarray,
    token_str: np.ndarray,
    vectors: np.ndarray,
    targets: Mapping[str, tuple[str, ...]],
    doc_mask: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Mean of all matching token vectors for each target name.

    A target matches any surface string in `targets[name]`. If `doc_mask` is
    given, only rows whose document index has mask True are used.
    """
    dim = int(vectors.shape[1]) if getattr(vectors, "ndim", 0) == 2 else 0
    zero = {name: np.zeros(dim, dtype=np.float64) for name in targets}

    keep = _row_mask(np.asarray(doc_idx, dtype=np.int32), doc_mask)
    if not np.any(keep):
        return zero

    idx = np.flatnonzero(keep)
    toks = np.asarray(token_str, dtype=object)[idx]
    vecs = np.asarray(vectors, dtype=np.float64)[idx]

    # Build term -> target-name list once; then vectorised gather per target.
    term_to_names: dict[str, list[str]] = {}
    for name, terms in targets.items():
        for term in terms:
            term_to_names.setdefault(str(term), []).append(name)

    # Invert: target name -> set of matching surface strings
    name_terms: dict[str, set[str]] = {name: set() for name in targets}
    for term, names in term_to_names.items():
        for name in names:
            name_terms[name].add(term)

    # String-compare once into a boolean matrix is costly; per-target mask is fine
    # for tens of targets and ~1e5 rows (permutation path).
    toks_str = np.array([str(t) for t in toks], dtype=object)
    out: dict[str, np.ndarray] = {}
    for name, term_set in name_terms.items():
        if not term_set:
            out[name] = np.zeros(dim, dtype=np.float64)
            continue
        mask = np.fromiter((t in term_set for t in toks_str), dtype=bool, count=len(toks_str))
        if not np.any(mask):
            out[name] = np.zeros(dim, dtype=np.float64)
        else:
            out[name] = vecs[mask].mean(axis=0)
    return out


def type_embeddings(
    doc_idx: np.ndarray,
    token_str: np.ndarray,
    vectors: np.ndarray,
    vocabulary: Sequence[str],
    doc_mask: np.ndarray | None = None,
) -> tuple[tuple[str, ...], np.ndarray]:
    """Mean vector per vocabulary type that is present under the doc mask.

    Returns (names_tuple, matrix) with rows aligned to names. Types with no
    occurrences are omitted (not zero-filled), so neighbour pools stay honest.
    """
    dim = int(vectors.shape[1]) if getattr(vectors, "ndim", 0) == 2 else 0
    wanted = {str(w) for w in vocabulary}
    keep = _row_mask(np.asarray(doc_idx, dtype=np.int32), doc_mask)
    if not np.any(keep):
        return tuple(), np.zeros((0, dim), dtype=np.float64)

    idx = np.flatnonzero(keep)
    toks = np.asarray(token_str, dtype=object)[idx]
    vecs = np.asarray(vectors, dtype=np.float64)[idx]

    # Filter to vocabulary members, then group by unique token string.
    keys = np.array([str(t) for t in toks], dtype=object)
    in_vocab = np.fromiter((k in wanted for k in keys), dtype=bool, count=len(keys))
    if not np.any(in_vocab):
        return tuple(), np.zeros((0, dim), dtype=np.float64)
    keys = keys[in_vocab]
    vecs = vecs[in_vocab]

    # Sort + reduce: contiguous means per unique key.
    order = np.argsort(keys, kind="mergesort")
    keys_s = keys[order]
    vecs_s = vecs[order]
    # Boundaries where key changes
    change = np.empty(len(keys_s), dtype=bool)
    change[0] = True
    change[1:] = keys_s[1:] != keys_s[:-1]
    starts = np.flatnonzero(change)
    names_list = [str(keys_s[i]) for i in starts]
    means = []
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(keys_s)
        means.append(vecs_s[start:end].mean(axis=0))
    names = tuple(names_list)  # already sorted by argsort
    matrix = np.vstack(means)
    return names, matrix


def neighbors_from_sense(
    sense: np.ndarray,
    type_names: Sequence[str],
    type_matrix: np.ndarray,
    k: int,
    exclude: Iterable[str] = (),
) -> list[tuple[str, float]]:
    """Top-k type names by cosine to the sense vector."""
    if len(type_names) == 0 or type_matrix.size == 0:
        return []
    sense_arr = np.asarray(sense, dtype=np.float64).ravel()
    types = np.asarray(type_matrix, dtype=np.float64)
    if types.ndim != 2 or types.shape[0] != len(type_names):
        raise ValueError(
            f"type_matrix shape {types.shape} does not match {len(type_names)} names"
        )
    if np.linalg.norm(sense_arr) == 0:
        return []
    sense_u = l2_normalize(sense_arr.reshape(1, -1))[0]
    types_u = l2_normalize(types)
    sims = types_u @ sense_u
    return top_k_neighbors(sims.tolist(), type_names, k=k, exclude=exclude)


def all_but_the_top(
    profiles: Mapping[str, np.ndarray], n_components: int
) -> dict[str, np.ndarray]:
    """Mean-centre and project out the top principal components (Mu & Viswanath).

    Returns a new dict; input is not mutated. If there are too few profiles to
    estimate `n_components`, returns copies of the originals.
    """
    if n_components <= 0:
        return {name: np.asarray(vec, dtype=np.float64).copy() for name, vec in profiles.items()}
    names = sorted(profiles)
    if not names:
        return {}
    matrix = np.vstack([np.asarray(profiles[n], dtype=np.float64) for n in names])
    if matrix.shape[0] <= n_components:
        return {name: matrix[i].copy() for i, name in enumerate(names)}
    centred = matrix - matrix.mean(axis=0, keepdims=True)
    from sklearn.decomposition import PCA

    n_comp = min(n_components, centred.shape[0] - 1, centred.shape[1])
    if n_comp < 1:
        return {name: centred[i].copy() for i, name in enumerate(names)}
    pca = PCA(n_components=n_comp, random_state=0)
    pca.fit(centred)
    corrected = centred - (centred @ pca.components_.T) @ pca.components_
    return {name: corrected[i] for i, name in enumerate(names)}


def all_but_the_top_matrix(
    matrix: np.ndarray, n_components: int
) -> np.ndarray:
    """ABTT on an (n, dim) matrix of type/sense vectors. Returns a copy."""
    array = np.asarray(matrix, dtype=np.float64)
    if n_components <= 0 or array.ndim != 2 or array.shape[0] <= n_components:
        return array.copy()
    centred = array - array.mean(axis=0, keepdims=True)
    from sklearn.decomposition import PCA

    n_comp = min(n_components, centred.shape[0] - 1, centred.shape[1])
    if n_comp < 1:
        return centred.copy()
    pca = PCA(n_components=n_comp, random_state=0)
    pca.fit(centred)
    return centred - (centred @ pca.components_.T) @ pca.components_


def count_target_freq(
    token_str: np.ndarray,
    doc_idx: np.ndarray,
    terms: Iterable[str],
    doc_mask: np.ndarray | None = None,
) -> int:
    """Count cached token rows matching any of `terms` (optionally masked)."""
    term_set = {str(t) for t in terms}
    keep = _row_mask(np.asarray(doc_idx, dtype=np.int32), doc_mask)
    toks = np.asarray(token_str, dtype=object)
    n = 0
    for i in np.flatnonzero(keep):
        if str(toks[i]) in term_set:
            n += 1
    return n


def n_docs_from_cache(doc_idx: np.ndarray) -> int:
    """Number of documents implied by the max doc index (+1), or 0 if empty."""
    if len(doc_idx) == 0:
        return 0
    return int(np.max(doc_idx)) + 1
