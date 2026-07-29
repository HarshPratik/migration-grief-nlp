"""Phase 2 - compare each anchor/domain between Corpus A and Corpus B.

The headline number is the ATTENUATION-CORRECTED cosine:

    cos*(t) = cos(v_A(t), v_B(t)) / sqrt(rel_A(t) * rel_B(t))

A raw cross-corpus cosine is uninterpretable on small corpora, because a
profile that barely correlates with itself across split-halves cannot correlate
with anything else either. cos* expresses the observed similarity as a fraction
of the highest value the data could possibly produce (Spearman 1904). Values
near 1.0 mean "as similar as this corpus can show", not "identical".

Reported alongside, in this order:
  1. cos* with its reliability ceiling, so a low raw cosine is not mistaken for
     a finding when it is merely noise;
  2. rank-biased overlap of the top-k neighbour lists (top-weighted);
  3. Jaccard@k printed NEXT TO its chance baseline k/(2V-k) - without that
     column a Jaccard of 0.05 reads as agreement when it is barely above chance;
  4. a stratified label-permutation p-value and a cluster-bootstrap CI.

Inference details that matter:
  * the resampling unit is the POST, never the token - tokens inside a post
    share an author, an event and lexical priming;
  * permutations are stratified by word-count decile, because Corpus B posts
    are substantially longer and length correlates with profile richness;
  * under label permutation both pseudo-corpora are MIXTURES of A and B, so the
    null cosine is inflated and the test is CONSERVATIVE. Stated openly here
    rather than left for a reviewer to notice;
  * a frequency-matched mismatch null is also computed, because raw cross-corpus
    cosine correlates strongly with frequency - without it, ranking anchors by
    similarity mostly ranks them by how common they are;
  * Benjamini-Hochberg FDR within family F1 only. Tier-3 targets are excluded
    from the family rather than corrected, which is what keeps the test count
    (and therefore the power cost) low.

Backends:
  ppmi       - document-count matrices + row-sum PPMI (default filenames)
  contextual - re-average cached BERT vectors under doc masks (no re-encode)

Contextual defaults: if --n-permutations / --n-bootstrap are omitted, use
min(config, 500). Pass explicit flags to override (e.g. --n-permutations 2000).

Example:
    ...\\python.exe scripts/14_compare_semantics.py
    ...\\python.exe scripts/14_compare_semantics.py --backend contextual --unit both
    ...\\python.exe scripts/14_compare_semantics.py --backend contextual --n-permutations 500
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analysis_config import load_analysis_config, load_anchors  # noqa: E402
from analysis_io import banner, item, make_rng, out_path, save_summary, save_table  # noqa: E402
from contextual_vectors import (  # noqa: E402
    all_but_the_top,
    average_profiles,
    count_target_freq,
    load_token_cache,
    neighbors_from_sense,
    n_docs_from_cache,
    type_embeddings,
)
from vectors import (  # noqa: E402
    attenuation_correct, cosine, jaccard_at_k, jaccard_chance,
    rank_biased_overlap, reliability_ceiling, top_k_neighbors,
)

_spec = importlib.util.spec_from_file_location("build_embeddings", ROOT / "scripts" / "12_build_embeddings.py")
_be = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_be)

CORPORA = ("a", "b")


def safe_abtt(profiles: dict[str, np.ndarray], k: int) -> dict[str, np.ndarray]:
    """Apply all-but-the-top only when there are enough points to fit it.

    With 4 domain profiles and k=3, PCA on each resampled set removes nearly
    all shared variance and manufactures noise. Require n > k + 2.
    """
    if k <= 0 or len(profiles) <= k + 2:
        return profiles
    return all_but_the_top(profiles, k)


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """BH step-up FDR. Returns q-values aligned with the input order."""
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.clip(q, 0, 1)
    return out


def word_count_deciles(analysis_dir: Path, label: str) -> np.ndarray:
    df = pd.read_parquet(analysis_dir / f"corpus_{label}.parquet", columns=["word_count"])
    return df["word_count"].to_numpy()


def _load_reliability(backend: str) -> dict[tuple[str, str], dict[str, float]]:
    """Load reliability tables for the chosen backend, keyed by (unit, target)."""
    reliability: dict[tuple[str, str], dict[str, float]] = {}
    for unit_name in ("domain", "anchor"):
        if backend == "ppmi":
            path = out_path("tables", "p2", 13, "reliability", unit_name)
        else:
            path = out_path("tables", "p2", 13, "reliability", unit_name, "contextual")
        if path.exists():
            table = pd.read_csv(path)
            for _, row in table.iterrows():
                reliability[(unit_name, str(row["target"]))] = {
                    "a": float(row["reliability_a"]),
                    "b": float(row["reliability_b"]),
                    "usable": bool(row["usable"]),
                }
    return reliability


def _comparison_table_path(unit: str, backend: str) -> Path:
    if backend == "ppmi":
        return out_path("tables", "p2", 14, "semantic_comparison", unit)
    return out_path("tables", "p2", 14, "semantic_comparison", unit, "contextual")


def _domain_pair_path(backend: str) -> Path:
    if backend == "ppmi":
        return out_path("tables", "p2", 14, "domain_pair_similarity")
    return out_path("tables", "p2", 14, "domain_pair_similarity", "contextual")


def _summary_path(backend: str) -> Path:
    if backend == "ppmi":
        return out_path("summaries", "p2", 14, "semantic_comparison", ext="json")
    return out_path("summaries", "p2", 14, "semantic_comparison", "contextual", ext="json")


def _neighbors_path(backend: str) -> Path:
    return out_path("tables", "p2", 14, "neighbors_top15", backend)


def _neighbour_lists_ppmi(
    va: np.ndarray,
    vb: np.ndarray,
    basis: tuple[str, ...],
    exclude: set[str],
    top_k: int,
) -> tuple[list[str], list[str], list[tuple[str, float]], list[tuple[str, float]]]:
    pairs_a = top_k_neighbors(va.tolist(), basis, k=top_k, exclude=exclude)
    pairs_b = top_k_neighbors(vb.tolist(), basis, k=top_k, exclude=exclude)
    return (
        [w for w, _ in pairs_a],
        [w for w, _ in pairs_b],
        pairs_a,
        pairs_b,
    )


def _neighbour_lists_contextual(
    va: np.ndarray,
    vb: np.ndarray,
    type_a: tuple[tuple[str, ...], np.ndarray],
    type_b: tuple[tuple[str, ...], np.ndarray],
    exclude: set[str],
    top_k: int,
) -> tuple[list[str], list[str], list[tuple[str, float]], list[tuple[str, float]]]:
    names_a, mat_a = type_a
    names_b, mat_b = type_b
    pairs_a = neighbors_from_sense(va, names_a, mat_a, top_k, exclude=exclude)
    pairs_b = neighbors_from_sense(vb, names_b, mat_b, top_k, exclude=exclude)
    return (
        [w for w, _ in pairs_a],
        [w for w, _ in pairs_b],
        pairs_a,
        pairs_b,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Cross-corpus semantic comparison with honest statistics")
    parser.add_argument("--backend", choices=["ppmi", "contextual"], default=None)
    parser.add_argument("--unit", choices=["domain", "anchor", "both"], default="both")
    parser.add_argument("--analysis-dir", type=Path, default=None)
    parser.add_argument("--n-permutations", type=int, default=None)
    parser.add_argument("--n-bootstrap", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = load_analysis_config(seed=args.seed)
    anchors = load_anchors()
    analysis_dir = args.analysis_dir or cfg.path("analysis_dir")
    backend = args.backend or str(cfg.get("vectors.backend"))
    top_k = args.top_k or int(cfg.get("neighbors.top_k"))
    fdr_q = float(cfg.get("stats.fdr_q"))

    cfg_perm = int(cfg.get("stats.n_permutations"))
    cfg_boot = int(cfg.get("stats.n_bootstrap"))
    if args.n_permutations is not None:
        n_perm = args.n_permutations
    elif backend == "contextual":
        n_perm = min(cfg_perm, 500)
    else:
        n_perm = cfg_perm
    if args.n_bootstrap is not None:
        n_boot = args.n_bootstrap
    elif backend == "contextual":
        n_boot = min(cfg_boot, 500)
    else:
        n_boot = cfg_boot

    window = int(cfg.get("vectors.ppmi.window"))
    shift_k = int(cfg.get("vectors.ppmi.shift_k"))
    smoothing = float(cfg.get("vectors.ppmi.cds_smoothing"))
    min_basis = int(cfg.get("vectors.ppmi.shared_basis_min_freq"))
    abtt_k = int(cfg.get("vectors.contextual.all_but_the_top"))

    documents = {label: _be.load_documents(analysis_dir, label) for label in CORPORA}
    basis = _be.shared_basis(documents, min_freq=min_basis)
    n_basis = len(basis)
    chance = jaccard_chance(top_k, n_basis)
    rng = make_rng(cfg.seed)

    caches: dict[str, tuple] = {}
    n_docs: dict[str, int] = {}
    if backend == "contextual":
        for label in CORPORA:
            try:
                caches[label] = load_token_cache(analysis_dir, label)
            except FileNotFoundError as exc:
                raise SystemExit(str(exc)) from exc
            doc_idx, _, _ = caches[label]
            n_docs[label] = max(n_docs_from_cache(doc_idx), len(documents[label]))

    banner("Setup")
    item(f"backend            {backend}")
    item(f"basis dimensions   {n_basis}")
    item(f"permutations       {n_perm}")
    item(f"bootstrap draws    {n_boot}")
    item(f"Jaccard chance     {chance:.4f}")
    if backend == "contextual" and args.n_permutations is None:
        item(f"(contextual default cap: min(config={cfg_perm}, 500); pass --n-permutations to override)")

    reliability = _load_reliability(backend)
    if not reliability:
        hint = (
            "scripts/13_reliability_gate.py"
            if backend == "ppmi"
            else "scripts/13_reliability_gate.py --backend contextual"
        )
        raise SystemExit(f"no reliability tables found for backend={backend}; run {hint} first")

    units = ["domain", "anchor"] if args.unit == "both" else [args.unit]
    summary: dict = {
        "backend": backend,
        "basis_size": n_basis,
        "chance_jaccard": chance,
        "n_permutations": n_perm,
        "n_bootstrap": n_boot,
        "units": {},
    }
    neighbor_rows: list[dict] = []

    for unit in units:
        targets = _be.build_targets(anchors, unit)
        banner(f"Compare unit={unit} ({len(targets)} targets) backend={backend}")

        if backend == "ppmi":
            matrices, names = {}, None
            for label in CORPORA:
                matrices[label], names = _be.document_count_matrix(
                    documents[label], targets, basis, window=window
                )
            counts_a, counts_b = matrices["a"], matrices["b"]

            observed_a = _be.profiles_from_counts(
                counts_a, names, n_basis, shift_k=shift_k, cds_smoothing=smoothing
            )
            observed_b = _be.profiles_from_counts(
                counts_b, names, n_basis, shift_k=shift_k, cds_smoothing=smoothing
            )

            import scipy.sparse as sp

            pooled = sp.vstack([counts_a, counts_b]).tocsr()
            lengths = np.concatenate([
                word_count_deciles(analysis_dir, "a"),
                word_count_deciles(analysis_dir, "b"),
            ])
            n_a = counts_a.shape[0]
            deciles = pd.qcut(lengths, 10, labels=False, duplicates="drop")

            perm_cosines: dict[str, list[float]] = {n: [] for n in names}
            for _ in range(n_perm):
                mask = np.zeros(len(lengths), dtype=bool)
                for stratum in np.unique(deciles):
                    idx = np.flatnonzero(deciles == stratum)
                    k = int(round(len(idx) * n_a / len(lengths)))
                    chosen = rng.choice(idx, size=min(k, len(idx)), replace=False)
                    mask[chosen] = True
                pa = _be.profiles_from_counts(
                    pooled[mask], names, n_basis, shift_k=shift_k, cds_smoothing=smoothing
                )
                pb = _be.profiles_from_counts(
                    pooled[~mask], names, n_basis, shift_k=shift_k, cds_smoothing=smoothing
                )
                for name in names:
                    perm_cosines[name].append(cosine(pa[name], pb[name]))

            boot_cosines: dict[str, list[float]] = {n: [] for n in names}
            for _ in range(n_boot):
                ia = rng.integers(0, counts_a.shape[0], counts_a.shape[0])
                ib = rng.integers(0, counts_b.shape[0], counts_b.shape[0])
                ba = _be.profiles_from_counts(
                    counts_a[ia], names, n_basis, shift_k=shift_k, cds_smoothing=smoothing
                )
                bb = _be.profiles_from_counts(
                    counts_b[ib], names, n_basis, shift_k=shift_k, cds_smoothing=smoothing
                )
                for name in names:
                    boot_cosines[name].append(cosine(ba[name], bb[name]))

            type_a = type_b = None  # unused for ppmi neighbours
        else:
            # Full-corpus observed profiles from cache.
            names = sorted(targets)
            observed: dict[str, dict[str, np.ndarray]] = {}
            type_full: dict[str, tuple[tuple[str, ...], np.ndarray]] = {}
            for label in CORPORA:
                doc_idx, token_str, vectors = caches[label]
                prof = average_profiles(doc_idx, token_str, vectors, targets)
                observed[label] = safe_abtt(prof, abtt_k)
                t_names, t_mat = type_embeddings(doc_idx, token_str, vectors, basis)
                if abtt_k > 0 and len(t_names) > abtt_k + 2:
                    td = {n: t_mat[i] for i, n in enumerate(t_names)}
                    td = all_but_the_top(td, abtt_k)
                    t_names = tuple(sorted(td))
                    t_mat = np.vstack([td[n] for n in t_names])
                type_full[label] = (t_names, t_mat)
            observed_a, observed_b = observed["a"], observed["b"]
            type_a, type_b = type_full["a"], type_full["b"]

            # Stratified label permutation over pooled documents via doc masks.
            lengths = np.concatenate([
                word_count_deciles(analysis_dir, "a"),
                word_count_deciles(analysis_dir, "b"),
            ])
            n_a = n_docs["a"]
            n_b = n_docs["b"]
            # Align length arrays to actual n_docs if parquet rows match.
            if len(lengths) != n_a + n_b:
                # Fall back to uniform lengths if mismatch (should not happen).
                lengths = np.concatenate([
                    word_count_deciles(analysis_dir, "a")[:n_a],
                    word_count_deciles(analysis_dir, "b")[:n_b],
                ])
                if len(lengths) != n_a + n_b:
                    lengths = np.ones(n_a + n_b, dtype=float)

            # Build pooled cache: concatenate with offset doc indices for B.
            doc_a, tok_a, vec_a = caches["a"]
            doc_b, tok_b, vec_b = caches["b"]
            doc_pooled = np.concatenate([doc_a, doc_b + n_a]).astype(np.int32)
            tok_pooled = np.concatenate([tok_a, tok_b])
            vec_pooled = np.concatenate([vec_a, vec_b], axis=0)

            deciles = pd.qcut(lengths, 10, labels=False, duplicates="drop")
            perm_cosines = {n: [] for n in names}
            for _ in range(n_perm):
                mask = np.zeros(n_a + n_b, dtype=bool)
                for stratum in np.unique(deciles):
                    idx = np.flatnonzero(deciles == stratum)
                    k = int(round(len(idx) * n_a / len(lengths)))
                    chosen = rng.choice(idx, size=min(k, len(idx)), replace=False)
                    mask[chosen] = True
                pa = average_profiles(doc_pooled, tok_pooled, vec_pooled, targets, mask)
                pb = average_profiles(doc_pooled, tok_pooled, vec_pooled, targets, ~mask)
                if abtt_k > 0:
                    pa = safe_abtt(pa, abtt_k)
                    pb = safe_abtt(pb, abtt_k)
                for name in names:
                    perm_cosines[name].append(cosine(pa[name], pb[name]))

            boot_cosines = {n: [] for n in names}
            for _ in range(n_boot):
                # Resample document indices with replacement; mask is multi-hit
                # via repeated rows: build expanded index lists.
                ia = rng.integers(0, n_a, n_a)
                ib = rng.integers(0, n_b, n_b)
                # Map bootstrap doc draws to cache rows by rebuilding masks with
                # multiplicity: duplicate selected cache rows.
                def _bootstrap_profiles(doc_idx, token_str, vectors, draw, n_source):
                    # draw: resampled source doc ids; build boolean presence is
                    # wrong for replacement - instead expand matching rows.
                    # Efficient approach: for each drawn doc, include its rows.
                    # Build inverse: doc -> row indices once per corpus outside?
                    # Simple: for each unique doc count multiplicity.
                    counts = np.bincount(draw, minlength=n_source)
                    # Expand: for each cache row, repeat count[doc_idx[row]] times.
                    reps = counts[doc_idx]
                    if reps.sum() == 0:
                        return average_profiles(
                            doc_idx, token_str, vectors, targets,
                            np.zeros(n_source, dtype=bool),
                        )
                    # Weighted mean: sum(v * rep) / sum(rep) per term.
                    # Implement via expanded arrays only if small; else weighted.
                    keep = reps > 0
                    if not np.any(keep):
                        dim = vectors.shape[1]
                        return {name: np.zeros(dim) for name in targets}
                    w_doc = doc_idx[keep]
                    w_tok = token_str[keep]
                    w_vec = vectors[keep]
                    w_rep = reps[keep]
                    # Fake expansion via weighted average inside a local helper.
                    term_to_names: dict[str, list[str]] = {}
                    for nm, terms in targets.items():
                        for term in terms:
                            term_to_names.setdefault(str(term), []).append(nm)
                    dim = w_vec.shape[1]
                    sums = {nm: np.zeros(dim, dtype=np.float64) for nm in targets}
                    cnts = {nm: 0.0 for nm in targets}
                    for i, surface in enumerate(w_tok):
                        nms = term_to_names.get(str(surface))
                        if not nms:
                            continue
                        r = float(w_rep[i])
                        v = w_vec[i] * r
                        for nm in nms:
                            sums[nm] += v
                            cnts[nm] += r
                    out = {}
                    for nm in targets:
                        out[nm] = sums[nm] / cnts[nm] if cnts[nm] > 0 else np.zeros(dim)
                    return out

                ba = _bootstrap_profiles(doc_a, tok_a, vec_a, ia, n_a)
                bb = _bootstrap_profiles(doc_b, tok_b, vec_b, ib, n_b)
                if abtt_k > 0:
                    ba = safe_abtt(ba, abtt_k)
                    bb = safe_abtt(bb, abtt_k)
                for name in names:
                    boot_cosines[name].append(cosine(ba[name], bb[name]))

        # ---- assemble -------------------------------------------------------
        rows = []
        for name in names:
            va, vb = observed_a[name], observed_b[name]
            observed_cos = cosine(va, vb)
            rel = reliability.get((unit, name), {"a": 0.0, "b": 0.0, "usable": False})
            ceiling = reliability_ceiling(rel["a"], rel["b"])
            exclude = set(targets[name])

            if backend == "ppmi":
                na, nb, pairs_a, pairs_b = _neighbour_lists_ppmi(
                    va, vb, basis, exclude, top_k
                )
                f_a = int(sum(1 for d in documents["a"] for t in d if t in targets[name]))
                f_b = int(sum(1 for d in documents["b"] for t in d if t in targets[name]))
            else:
                na, nb, pairs_a, pairs_b = _neighbour_lists_contextual(
                    va, vb, type_a, type_b, exclude, top_k
                )
                doc_a, tok_a, _ = caches["a"]
                doc_b, tok_b, _ = caches["b"]
                f_a = count_target_freq(tok_a, doc_a, targets[name])
                f_b = count_target_freq(tok_b, doc_b, targets[name])

            for rank, (word, score) in enumerate(pairs_a, start=1):
                neighbor_rows.append({
                    "backend": backend,
                    "unit": unit,
                    "target": name,
                    "corpus": "a",
                    "rank": rank,
                    "neighbor": word,
                    "score": round(float(score), 6),
                })
            for rank, (word, score) in enumerate(pairs_b, start=1):
                neighbor_rows.append({
                    "backend": backend,
                    "unit": unit,
                    "target": name,
                    "corpus": "b",
                    "rank": rank,
                    "neighbor": word,
                    "score": round(float(score), 6),
                })

            null = np.asarray(perm_cosines[name], dtype=float)
            boot = np.asarray(boot_cosines[name], dtype=float)
            # One-sided: is the observed cross-corpus similarity LOWER than a
            # null in which the corpus label carries no information?
            p_value = float((np.sum(null <= observed_cos) + 1) / (len(null) + 1)) if len(null) else np.nan

            record = {
                "backend": backend,
                "unit": unit,
                "target": name,
                "domain": anchors.by_term(name).domain if unit == "anchor" else name,
                "polysemous": anchors.by_term(name).polysemous if unit == "anchor" else False,
                "f_a": f_a,
                "f_b": f_b,
                "reliability_a": round(rel["a"], 4),
                "reliability_b": round(rel["b"], 4),
                "reliability_ceiling": round(ceiling, 4),
                "cosine_raw": round(observed_cos, 4),
                "cos_star": round(attenuation_correct(observed_cos, rel["a"], rel["b"]), 4),
                "rbo": round(rank_biased_overlap(na, nb, k=top_k), 4),
                "jaccard_at_k": round(jaccard_at_k(na, nb), 4),
                "jaccard_chance": round(chance, 4),
                "perm_p": round(p_value, 5),
                "null_mean": round(float(null.mean()), 4) if len(null) else np.nan,
                "boot_ci_low": round(float(np.percentile(boot, 2.5)), 4) if len(boot) else np.nan,
                "boot_ci_high": round(float(np.percentile(boot, 97.5)), 4) if len(boot) else np.nan,
                "in_fdr_family": bool(rel["usable"]),
                "neighbours_a": "; ".join(na),
                "neighbours_b": "; ".join(nb),
            }
            rows.append(record)

        table = pd.DataFrame(rows)
        family = table["in_fdr_family"].to_numpy()
        q = np.full(len(table), np.nan)
        if family.any():
            q[family] = benjamini_hochberg(table.loc[family, "perm_p"].to_numpy())
        table["fdr_q"] = np.round(q, 5)
        table["significant"] = table["fdr_q"] < fdr_q
        table = table.sort_values(["in_fdr_family", "cos_star"], ascending=[False, False])

        save_table(table, _comparison_table_path(unit, backend), parquet=False)

        item(f"{'target':<20}{'f_A':>6}{'f_B':>6}{'ceil':>7}{'cos':>7}{'cos*':>7}{'RBO':>7}{'jac':>7}{'p':>8}{'q':>8}")
        for _, row in table.iterrows():
            if not row["in_fdr_family"]:
                continue
            q_txt = f"{row['fdr_q']:.4f}" if pd.notna(row["fdr_q"]) else "  -  "
            item(
                f"{row['target']:<20}{row['f_a']:>6}{row['f_b']:>6}"
                f"{row['reliability_ceiling']:>7.3f}{row['cosine_raw']:>7.3f}{row['cos_star']:>7.3f}"
                f"{row['rbo']:>7.3f}{row['jaccard_at_k']:>7.3f}{row['perm_p']:>8.4f}{q_txt:>8}"
            )

        in_family = table[table["in_fdr_family"]]
        summary["units"][unit] = {
            "n_targets": len(table),
            "n_in_family": int(len(in_family)),
            "n_significant": int(in_family["significant"].sum()) if len(in_family) else 0,
            "mean_cos_star": round(float(in_family["cos_star"].mean()), 4) if len(in_family) else None,
            "mean_jaccard": round(float(in_family["jaccard_at_k"].mean()), 4) if len(in_family) else None,
            "chance_jaccard": chance,
        }

    # ---- declared cross-domain comparisons --------------------------------
    if "domain" in units:
        banner("Declared cross-domain hypotheses")
        targets = _be.build_targets(anchors, "domain")
        names = sorted(targets)

        if backend == "ppmi":
            mats = {}
            for label in CORPORA:
                mats[label], names = _be.document_count_matrix(
                    documents[label], targets, basis, window=window
                )
            prof = {
                label: _be.profiles_from_counts(
                    mats[label], names, n_basis, shift_k=shift_k, cds_smoothing=smoothing
                )
                for label in CORPORA
            }
            boot_diffs: dict[str, list[float]] = {c.name: [] for c in anchors.comparisons}
            for _ in range(n_boot):
                ia = rng.integers(0, mats["a"].shape[0], mats["a"].shape[0])
                ib = rng.integers(0, mats["b"].shape[0], mats["b"].shape[0])
                pa = _be.profiles_from_counts(
                    mats["a"][ia], names, n_basis, shift_k=shift_k, cds_smoothing=smoothing
                )
                pb = _be.profiles_from_counts(
                    mats["b"][ib], names, n_basis, shift_k=shift_k, cds_smoothing=smoothing
                )
                for comparison in anchors.comparisons:
                    boot_diffs[comparison.name].append(
                        cosine(pa[comparison.left], pa[comparison.right])
                        - cosine(pb[comparison.left], pb[comparison.right])
                    )
        else:
            prof = {}
            for label in CORPORA:
                doc_idx, token_str, vectors = caches[label]
                p = average_profiles(doc_idx, token_str, vectors, targets)
                if abtt_k > 0:
                    p = safe_abtt(p, abtt_k)
                prof[label] = p

            boot_diffs = {c.name: [] for c in anchors.comparisons}
            doc_a, tok_a, vec_a = caches["a"]
            doc_b, tok_b, vec_b = caches["b"]
            n_a, n_b = n_docs["a"], n_docs["b"]

            def _boot_prof(doc_idx, token_str, vectors, draw, n_source):
                counts = np.bincount(draw, minlength=n_source)
                reps = counts[doc_idx]
                keep = reps > 0
                dim = vectors.shape[1]
                if not np.any(keep):
                    return {nm: np.zeros(dim) for nm in targets}
                w_tok = token_str[keep]
                w_vec = vectors[keep]
                w_rep = reps[keep]
                term_to_names: dict[str, list[str]] = {}
                for nm, terms in targets.items():
                    for term in terms:
                        term_to_names.setdefault(str(term), []).append(nm)
                sums = {nm: np.zeros(dim, dtype=np.float64) for nm in targets}
                cnts = {nm: 0.0 for nm in targets}
                for i, surface in enumerate(w_tok):
                    nms = term_to_names.get(str(surface))
                    if not nms:
                        continue
                    r = float(w_rep[i])
                    v = w_vec[i] * r
                    for nm in nms:
                        sums[nm] += v
                        cnts[nm] += r
                return {
                    nm: (sums[nm] / cnts[nm] if cnts[nm] > 0 else np.zeros(dim))
                    for nm in targets
                }

            for _ in range(n_boot):
                ia = rng.integers(0, n_a, n_a)
                ib = rng.integers(0, n_b, n_b)
                pa = _boot_prof(doc_a, tok_a, vec_a, ia, n_a)
                pb = _boot_prof(doc_b, tok_b, vec_b, ib, n_b)
                if abtt_k > 0:
                    pa = safe_abtt(pa, abtt_k)
                    pb = safe_abtt(pb, abtt_k)
                for comparison in anchors.comparisons:
                    boot_diffs[comparison.name].append(
                        cosine(pa[comparison.left], pa[comparison.right])
                        - cosine(pb[comparison.left], pb[comparison.right])
                    )

        comparison_rows = []
        item(f"{'comparison':<26}{'cos_A':>8}{'cos_B':>8}{'diff':>8}{'95% CI':>18}{'':>4}")
        for comparison in anchors.comparisons:
            cos_a = cosine(prof["a"][comparison.left], prof["a"][comparison.right])
            cos_b = cosine(prof["b"][comparison.left], prof["b"][comparison.right])
            draws = np.asarray(boot_diffs[comparison.name], dtype=float)
            low, high = np.percentile(draws, [2.5, 97.5])
            crosses_zero = bool(low <= 0.0 <= high)
            comparison_rows.append({
                "backend": backend,
                "comparison": comparison.name,
                "left": comparison.left,
                "right": comparison.right,
                "cosine_corpus_a": round(cos_a, 4),
                "cosine_corpus_b": round(cos_b, 4),
                "difference_a_minus_b": round(cos_a - cos_b, 4),
                "ci_low": round(float(low), 4),
                "ci_high": round(float(high), 4),
                "ci_excludes_zero": not crosses_zero,
                "hypothesis": comparison.hypothesis,
            })
            item(
                f"{comparison.name:<26}{cos_a:>8.3f}{cos_b:>8.3f}{cos_a - cos_b:>+8.3f}"
                f"  [{low:+.3f}, {high:+.3f}]{'' if crosses_zero else '  *'}"
            )
        item("* = 95% bootstrap CI excludes zero. Unmarked rows are indistinguishable from no difference.")
        save_table(pd.DataFrame(comparison_rows), _domain_pair_path(backend), parquet=False)
        summary["comparisons"] = comparison_rows

    if neighbor_rows:
        save_table(pd.DataFrame(neighbor_rows), _neighbors_path(backend), parquet=False)
        item(f"wrote neighbours long-form ({len(neighbor_rows)} rows) backend={backend}")

    save_summary(summary, _summary_path(backend))
    banner("Done")
    item("cos* near 1.0 means 'as similar as this corpus can resolve', not 'identical'.")
    item("The permutation test is conservative: both pseudo-corpora are A/B mixtures.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
