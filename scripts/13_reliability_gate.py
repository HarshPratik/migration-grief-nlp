"""Phase 2 - GATE 2. Can this corpus measure these profiles at all?

A cross-corpus cosine is meaningless if a target's profile does not even
correlate with ITSELF when estimated from two halves of the same corpus. This
script measures that directly, and it is a go/no-go rather than a formality.

Method: split the documents of one corpus into random halves, build each
target's profile independently on each half, take the cosine between them, and
repeat. Splits are over DOCUMENTS, never tokens - tokens inside a post share an
author, an event and lexical priming, so token-level resampling understates
variance and inflates every downstream significance test.

The split-half value is projected to full-corpus reliability with the
Spearman-Brown formula, 2r/(1+r). That estimate feeds the attenuation
correction in script 14: cos* = cos(A,B) / sqrt(rel_A * rel_B).

Also measured: the split-half agreement of TOP-K NEIGHBOUR LISTS, reported
beside its chance baseline k/(2V-k). A prior run of this design on a smaller
snapshot of these corpora found neighbour agreement AT chance for every
frequency band below f=200. If that repeats, neighbour lists are not a finding
and belong in an appendix as a negative methodological result - which is itself
worth publishing, because small-corpus studies routinely skip this check.

Backends:
  ppmi       - rebuild PPMI profiles on each half (default; backward-compatible names)
  contextual - re-average cached BERT token vectors under document masks (no re-encode)

Example:
    ...\\python.exe scripts/13_reliability_gate.py
    ...\\python.exe scripts/13_reliability_gate.py --backend contextual --unit anchor --n-splits 25
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from collections import Counter
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
from vectors import cosine, jaccard_at_k, jaccard_chance, spearman_brown, top_k_neighbors  # noqa: E402

_spec = importlib.util.spec_from_file_location("build_embeddings", ROOT / "scripts" / "12_build_embeddings.py")
_be = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_be)

CORPORA = ("a", "b")


def neighbours_from_profile(
    profile: np.ndarray, basis: tuple[str, ...], *, k: int, exclude: set[str]
) -> list[str]:
    """Top-k basis dimensions by PPMI weight - the profile's 'neighbours'."""
    pairs = top_k_neighbors(profile.tolist(), basis, k=k, exclude=exclude)
    return [word for word, _ in pairs]


def _reliability_table_path(unit: str, backend: str) -> Path:
    if backend == "ppmi":
        return out_path("tables", "p2", 13, "reliability", unit)
    return out_path("tables", "p2", 13, "reliability", unit, "contextual")


def _reliability_summary_path(backend: str) -> Path:
    if backend == "ppmi":
        return out_path("summaries", "p2", 13, "reliability", ext="json")
    return out_path("summaries", "p2", 13, "reliability", "contextual", ext="json")


def main() -> int:
    parser = argparse.ArgumentParser(description="Split-half reliability gate for anchor/domain profiles")
    parser.add_argument("--backend", choices=["ppmi", "contextual"], default=None)
    parser.add_argument("--unit", choices=["domain", "anchor", "both"], default="both")
    parser.add_argument("--analysis-dir", type=Path, default=None)
    parser.add_argument("--n-splits", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-gate", action="store_true")
    args = parser.parse_args()

    cfg = load_analysis_config(seed=args.seed)
    anchors = load_anchors()
    analysis_dir = args.analysis_dir or cfg.path("analysis_dir")
    backend = args.backend or str(cfg.get("vectors.backend"))
    top_k = args.top_k or int(cfg.get("neighbors.top_k"))
    threshold = float(cfg.get("reliability.min_reliability_for_inference"))
    min_freq_floor = int(cfg.get("tiers.tier2.min_freq"))

    # Contextual defaults: fewer splits (each still cheap from cache, but type
    # embedding recompute dominates). Override with --n-splits.
    cfg_splits = int(cfg.get("reliability.n_splits"))
    if args.n_splits is not None:
        n_splits = args.n_splits
    elif backend == "contextual":
        n_splits = min(cfg_splits, 25)
    else:
        n_splits = cfg_splits

    window = int(cfg.get("vectors.ppmi.window"))
    shift_k = int(cfg.get("vectors.ppmi.shift_k"))
    smoothing = float(cfg.get("vectors.ppmi.cds_smoothing"))
    min_basis = int(cfg.get("vectors.ppmi.shared_basis_min_freq"))
    abtt_k = int(cfg.get("vectors.contextual.all_but_the_top"))

    documents = {label: _be.load_documents(analysis_dir, label) for label in CORPORA}
    basis = _be.shared_basis(documents, min_freq=min_basis)

    caches: dict[str, tuple] = {}
    if backend == "contextual":
        for label in CORPORA:
            try:
                caches[label] = load_token_cache(analysis_dir, label)
            except FileNotFoundError as exc:
                raise SystemExit(str(exc)) from exc
        # Neighbour chance uses types present on full corpus (approx basis size).
        vocab_size = len(basis)
        chance = jaccard_chance(top_k, vocab_size)
    else:
        chance = jaccard_chance(top_k, len(basis))

    banner("Reliability gate")
    item(f"backend                {backend}")
    item(f"splits per target       {n_splits}")
    item(f"basis dimensions        {len(basis)}")
    item(f"neighbour list depth    {top_k}")
    item(f"Jaccard chance baseline {chance:.4f}")
    item(f"usable threshold        Spearman-Brown >= {threshold} AND min freq >= {min_freq_floor}")

    units = ["domain", "anchor"] if args.unit == "both" else [args.unit]
    rng = make_rng(cfg.seed)
    summary: dict = {
        "backend": backend,
        "basis_size": len(basis),
        "n_splits": n_splits,
        "chance_jaccard": chance,
        "units": {},
    }
    all_rows: list[dict] = []

    for unit in units:
        targets = _be.build_targets(anchors, unit)
        names = sorted(targets)
        banner(f"unit={unit} ({len(names)} targets) backend={backend}")

        cos_samples: dict[str, dict[str, list[float]]] = {c: {n: [] for n in names} for c in CORPORA}
        jac_samples: dict[str, dict[str, list[float]]] = {c: {n: [] for n in names} for c in CORPORA}

        for label in CORPORA:
            if backend == "ppmi":
                docs = documents[label]
                n_docs = len(docs)
                for _ in range(n_splits):
                    order = rng.permutation(n_docs)
                    mid = n_docs // 2
                    first = [docs[i] for i in order[:mid]]
                    second = [docs[i] for i in order[mid:]]

                    left = _be.ppmi_profiles(
                        first, targets, basis, window=window, shift_k=shift_k, cds_smoothing=smoothing
                    )
                    right = _be.ppmi_profiles(
                        second, targets, basis, window=window, shift_k=shift_k, cds_smoothing=smoothing
                    )

                    for name in names:
                        lv, rv = left[name], right[name]
                        if np.linalg.norm(lv) == 0 or np.linalg.norm(rv) == 0:
                            continue
                        cos_samples[label][name].append(cosine(lv, rv))
                        exclude = set(targets[name])
                        jac_samples[label][name].append(
                            jaccard_at_k(
                                neighbours_from_profile(lv, basis, k=top_k, exclude=exclude),
                                neighbours_from_profile(rv, basis, k=top_k, exclude=exclude),
                            )
                        )
            else:
                doc_idx, token_str, vectors = caches[label]
                n_docs = max(n_docs_from_cache(doc_idx), len(documents[label]))
                for _ in range(n_splits):
                    order = rng.permutation(n_docs)
                    mid = n_docs // 2
                    mask_left = np.zeros(n_docs, dtype=bool)
                    mask_right = np.zeros(n_docs, dtype=bool)
                    mask_left[order[:mid]] = True
                    mask_right[order[mid:]] = True

                    # Do NOT apply all-but-the-top per half. With only 4 domain
                    # targets, fitting 3 PCs on each half projects almost all
                    # shared signal out and forces split-half cosine near 0 even
                    # when the raw mean-pooled senses are highly stable. ABTT is
                    # reserved for full-corpus profile export (script 12/14).
                    left = average_profiles(doc_idx, token_str, vectors, targets, mask_left)
                    right = average_profiles(doc_idx, token_str, vectors, targets, mask_right)

                    type_l_names, type_l_mat = type_embeddings(
                        doc_idx, token_str, vectors, basis, mask_left
                    )
                    type_r_names, type_r_mat = type_embeddings(
                        doc_idx, token_str, vectors, basis, mask_right
                    )

                    for name in names:
                        lv, rv = left[name], right[name]
                        if np.linalg.norm(lv) == 0 or np.linalg.norm(rv) == 0:
                            continue
                        cos_samples[label][name].append(cosine(lv, rv))
                        exclude = set(targets[name])
                        nl = [w for w, _ in neighbors_from_sense(
                            lv, type_l_names, type_l_mat, top_k, exclude=exclude
                        )]
                        nr = [w for w, _ in neighbors_from_sense(
                            rv, type_r_names, type_r_mat, top_k, exclude=exclude
                        )]
                        jac_samples[label][name].append(jaccard_at_k(nl, nr))

        # frequency, for the frequency-vs-reliability relationship
        freq: dict[str, Counter] = {}
        if backend == "ppmi":
            for label in CORPORA:
                counter: Counter = Counter()
                for tokens in documents[label]:
                    counter.update(tokens)
                freq[label] = counter

        rows = []
        for name in names:
            record: dict = {"unit": unit, "target": name, "backend": backend}
            if unit == "anchor":
                record["domain"] = anchors.by_term(name).domain
                record["polysemous"] = anchors.by_term(name).polysemous
            else:
                record["domain"] = name
                record["polysemous"] = False
            for label in CORPORA:
                terms = targets[name]
                if backend == "ppmi":
                    f = sum(freq[label].get(t, 0) for t in terms)
                else:
                    doc_idx, token_str, _ = caches[label]
                    f = count_target_freq(token_str, doc_idx, terms)
                cs = cos_samples[label][name]
                js = jac_samples[label][name]
                split_half = float(np.mean(cs)) if cs else 0.0
                record[f"f_{label}"] = f
                record[f"split_half_{label}"] = round(split_half, 4)
                record[f"reliability_{label}"] = round(spearman_brown(split_half), 4)
                record[f"neighbour_jaccard_{label}"] = round(float(np.mean(js)), 4) if js else 0.0
            record["min_reliability"] = round(min(record["reliability_a"], record["reliability_b"]), 4)
            record["min_f"] = min(record["f_a"], record["f_b"])
            # A frequency floor is required as well as a reliability floor.
            # At very low n the split-half cosine can be spuriously HIGH: with
            # 2 occurrences both halves may draw the same single context, so
            # the profile "agrees with itself" for a trivial reason. Observed
            # here on fit_in (f_B=2, rel 0.47) and whatsapp (f_A=2, rel 0.55).
            record["usable"] = bool(
                record["min_reliability"] >= threshold and record["min_f"] >= min_freq_floor
            )
            record["neighbours_above_chance"] = bool(
                min(record["neighbour_jaccard_a"], record["neighbour_jaccard_b"]) > 2 * chance
            )
            rows.append(record)

        table = pd.DataFrame(rows).sort_values("min_reliability", ascending=False)
        all_rows.extend(rows)
        save_table(table, _reliability_table_path(unit, backend), parquet=False)

        item(f"{'target':<24}{'f_A':>7}{'f_B':>7}{'relA':>7}{'relB':>7}{'jacA':>7}{'jacB':>7}  usable")
        for _, row in table.iterrows():
            flag = "yes" if row["usable"] else "NO"
            item(
                f"{row['target']:<24}{row['f_a']:>7}{row['f_b']:>7}"
                f"{row['reliability_a']:>7.3f}{row['reliability_b']:>7.3f}"
                f"{row['neighbour_jaccard_a']:>7.3f}{row['neighbour_jaccard_b']:>7.3f}  {flag}"
            )

        usable = int(table["usable"].sum())
        above_chance = int(table["neighbours_above_chance"].sum())
        summary["units"][unit] = {
            "n_targets": len(names),
            "n_usable": usable,
            "n_neighbours_above_chance": above_chance,
            "mean_reliability_a": round(float(table["reliability_a"].mean()), 4),
            "mean_reliability_b": round(float(table["reliability_b"].mean()), 4),
            "mean_neighbour_jaccard_a": round(float(table["neighbour_jaccard_a"].mean()), 4),
            "mean_neighbour_jaccard_b": round(float(table["neighbour_jaccard_b"].mean()), 4),
        }

    # ---- verdict ----------------------------------------------------------
    banner("Verdict")
    for unit, stats in summary["units"].items():
        item(f"{unit}: {stats['n_usable']}/{stats['n_targets']} targets reach reliability >= {threshold}")
        item(
            f"mean neighbour Jaccard A={stats['mean_neighbour_jaccard_a']:.4f} "
            f"B={stats['mean_neighbour_jaccard_b']:.4f} vs chance {chance:.4f}",
            indent=2,
        )
        item(f"{stats['n_neighbours_above_chance']}/{stats['n_targets']} targets have neighbour lists clearly above chance", indent=2)

    domain_stats = summary["units"].get("domain")
    anchor_stats = summary["units"].get("anchor")
    # Domain is preferred for the paper, but anchor-level BERT senses can clear
    # the gate when domain pooling does not. Either unit is enough to proceed.
    reportable = bool(
        (domain_stats and domain_stats["n_usable"] > 0)
        or (anchor_stats and anchor_stats["n_usable"] > 0)
    )
    summary["neighbour_lists_reportable"] = bool(
        (domain_stats and domain_stats.get("n_neighbours_above_chance", 0) > 0)
        or (anchor_stats and anchor_stats.get("n_neighbours_above_chance", 0) > 0)
    )
    summary["profiles_reportable"] = reportable

    if not summary["neighbour_lists_reportable"]:
        item("NEIGHBOUR LISTS ARE NOT REPORTABLE as findings at this corpus size.")
        item("Report them in an appendix as a negative methodological result.", indent=2)
    save_summary(summary, _reliability_summary_path(backend))

    if not reportable and not args.no_gate:
        banner("GATE FAILED")
        item("No target reaches the reliability threshold; downstream cos* would be noise.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
