"""Empirical Word2Vec / FastText evaluation on the analysis corpora.

Research purpose: measure whether *trained* predictive embeddings are stable
enough on the current sample size to support the Phase 2 claims. This is not
a rejection in advance — it produces the same reliability + A/B comparison
tables as PPMI/BERT so the paper can show what works.

Alignment: each half / corpus trains its own model. We compare SECOND-ORDER
profiles (cosine of the target to every shared-basis type) so spaces need not
be Procrustes-aligned.

Example:
    python scripts/25_predictive_eval.py --algorithms word2vec fasttext
    python scripts/25_predictive_eval.py --algorithms word2vec --n-splits 10 --n-bootstrap 20
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analysis_config import load_analysis_config, load_anchors  # noqa: E402
from analysis_io import banner, item, make_rng, out_path, save_summary, save_table  # noqa: E402
from predictive_vectors import (  # noqa: E402
    neighbors_from_kv,
    profiles_for_targets,
    second_order_profile,
    train_keyed_vectors,
)
from vectors import (  # noqa: E402
    attenuation_correct,
    cosine,
    jaccard_at_k,
    jaccard_chance,
    rank_biased_overlap,
    reliability_ceiling,
    spearman_brown,
)

# Reuse shared basis + targets from the PPMI builder
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "build_embeddings", ROOT / "scripts" / "12_build_embeddings.py"
)
_be = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_be)

CORPORA = ("a", "b")

# ---------------------------------------------------------------------------
# Parallel training pool.
#
# The ~200 gensim trainings per algorithm are the bottleneck and are
# independent given their document subsets. Each training keeps workers=1
# and its fixed seed, so results are bit-identical to the sequential version;
# parallelism is ONLY across models. Corpora are handed to each worker once
# via the pool initializer; job payloads carry index lists instead of data.
# ---------------------------------------------------------------------------
_WORKER_STATE: dict = {}


def _init_worker(docs_a, docs_b, extra_terms):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    _WORKER_STATE["a"] = docs_a
    _WORKER_STATE["b"] = docs_b
    _WORKER_STATE["extra_terms"] = list(extra_terms)


class ArrayKeyedVectors:
    """Minimal wv replacement rebuilt from plain arrays (pickle-safe, exact).

    Supports the operations used downstream: ``in``, ``len``, ``[token]``,
    ``get_vector`` and ``index_to_key``. ``extra`` holds vectors synthesised
    for out-of-vocabulary target terms (FastText subwords), matching what
    ``target_vector`` would obtain from the live model.
    """

    __slots__ = ("_words", "_matrix", "_index", "_extra")

    def __init__(self, words, matrix, extra=None):
        self._words = list(words)
        self._matrix = np.asarray(matrix)
        self._index = {w: i for i, w in enumerate(self._words)}
        self._extra = dict(extra or {})

    def __contains__(self, key):
        return key in self._index or key in self._extra

    def __len__(self):
        return len(self._words)

    def __getitem__(self, key):
        return self.get_vector(key)

    def get_vector(self, key):
        idx = self._index.get(key)
        if idx is not None:
            return self._matrix[idx]
        vec = self._extra.get(key)
        if vec is None:
            raise KeyError(key)
        return vec

    @property
    def index_to_key(self):
        return self._words

    @property
    def vectors(self):
        return self._matrix


def _train_worker(payload):
    tag, source, indices, algorithm, params = payload
    if source == "a":
        corpus = _WORKER_STATE["a"]
    elif source == "b":
        corpus = _WORKER_STATE["b"]
    else:
        corpus = _WORKER_STATE["a"] + _WORKER_STATE["b"]
    docs = list(corpus) if indices is None else [corpus[int(i)] for i in indices]
    try:
        kv = train_keyed_vectors(docs, algorithm=algorithm, **params)
    except ValueError:
        return tag, None
    words = list(kv.index_to_key)
    matrix = (
        np.stack([np.asarray(kv.get_vector(w), dtype=np.float32) for w in words])
        if words
        else np.zeros((0, params.get("vector_size", 100)), np.float32)
    )
    extra = {}
    for term in _WORKER_STATE["extra_terms"]:
        if term not in kv:
            try:
                extra[term] = np.asarray(kv.get_vector(term), dtype=np.float32)
            except KeyError:
                continue
    return tag, (words, matrix, extra)


def _reconstruct_kv(data):
    if data is None:
        return None
    words, matrix, extra = data
    return ArrayKeyedVectors(words, matrix, extra)


def run_trainings(jobs, max_workers: int, desc: str) -> dict:
    """Run independent trainings; returns {tag: ArrayKeyedVectors | None}."""
    total = len(jobs)
    results: dict = {}
    if max_workers <= 1 or total <= 1:
        for payload in jobs:
            tag, data = _train_worker(payload)
            results[tag] = _reconstruct_kv(data)
        return results
    item(f"dispatching {total} {desc} trainings to {max_workers} workers", indent=1)
    done = 0
    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_worker,
        initargs=(
            _WORKER_STATE["a"],
            _WORKER_STATE["b"],
            _WORKER_STATE["extra_terms"],
        ),
    ) as pool:
        futures = [pool.submit(_train_worker, payload) for payload in jobs]
        for future in as_completed(futures):
            tag, data = future.result()
            results[tag] = _reconstruct_kv(data)
            done += 1
            if done % 10 == 0 or done == total:
                item(f"trained {done}/{total} {desc}", indent=2)
    return results


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
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


def train_params(cfg, seed: int) -> dict:
    return dict(
        vector_size=int(cfg.get("vectors.predictive.vector_size", 100)),
        window=int(cfg.get("vectors.predictive.window", 5)),
        min_count=int(cfg.get("vectors.predictive.min_count", 5)),
        epochs=int(cfg.get("vectors.predictive.epochs", 20)),
        sg=int(cfg.get("vectors.predictive.sg", 1)),
        negative=int(cfg.get("vectors.predictive.negative", 5)),
        seed=int(seed),
        workers=int(cfg.get("vectors.predictive.workers", 1)),
    )


def count_freq(documents: list[list[str]], terms: tuple[str, ...]) -> int:
    want = set(terms)
    return sum(1 for doc in documents for t in doc if t in want)


def main() -> int:
    parser = argparse.ArgumentParser(description="Word2Vec/FastText reliability + A/B comparison")
    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=["word2vec", "fasttext"],
        default=["word2vec", "fasttext"],
    )
    parser.add_argument("--unit", choices=["domain", "anchor", "both"], default="both")
    parser.add_argument("--n-splits", type=int, default=12)
    parser.add_argument("--n-bootstrap", type=int, default=20)
    parser.add_argument("--n-permutations", type=int, default=30)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-gate", action="store_true")
    parser.add_argument(
        "--train-workers",
        type=int,
        default=max(2, min(12, (os.cpu_count() or 4) - 4)),
        help="parallel training processes; each training itself keeps workers=1 "
        "so results are identical to sequential execution",
    )
    args = parser.parse_args()

    cfg = load_analysis_config(seed=args.seed)
    anchors = load_anchors()
    analysis_dir = cfg.path("analysis_dir")
    top_k = args.top_k or int(cfg.get("neighbors.top_k"))
    threshold = float(cfg.get("reliability.min_reliability_for_inference"))
    min_freq_floor = int(cfg.get("tiers.tier2.min_freq"))
    min_basis = int(cfg.get("vectors.ppmi.shared_basis_min_freq"))
    fdr_q = float(cfg.get("stats.fdr_q"))
    params = train_params(cfg, cfg.seed)
    units = ["domain", "anchor"] if args.unit == "both" else [args.unit]

    documents = {label: _be.load_documents(analysis_dir, label) for label in CORPORA}
    basis = _be.shared_basis(documents, min_freq=min_basis)
    chance = jaccard_chance(top_k, len(basis))
    rng = make_rng(cfg.seed)

    # Hand corpora + all OOV-synthesis target terms to the training pool.
    _WORKER_STATE["a"] = documents["a"]
    _WORKER_STATE["b"] = documents["b"]
    _extra_terms: set[str] = set()
    for _u in units:
        for _terms in _be.build_targets(anchors, _u).values():
            _extra_terms.update(str(t) for t in _terms)
    _WORKER_STATE["extra_terms"] = sorted(_extra_terms)

    banner("Predictive embedding evaluation (empirical small-N check)")
    item(f"algorithms     {args.algorithms}")
    item(f"basis size     {len(basis)}")
    item(f"vector_size    {params['vector_size']}  window={params['window']}  "
         f"min_count={params['min_count']}  epochs={params['epochs']}  sg={params['sg']}")
    item(f"splits={args.n_splits}  bootstrap={args.n_bootstrap}  permutations={args.n_permutations}")
    item(f"train workers  {args.train_workers} (each training single-threaded, seeds fixed)")
    item(f"Jaccard chance {chance:.4f}")

    vectors_dir = analysis_dir / "vectors"
    vectors_dir.mkdir(parents=True, exist_ok=True)

    for algorithm in args.algorithms:
        banner(f"ALGORITHM = {algorithm}")
        algo_summary: dict = {
            "algorithm": algorithm,
            "basis_size": len(basis),
            "params": params,
            "n_splits": args.n_splits,
            "n_bootstrap": args.n_bootstrap,
            "n_permutations": args.n_permutations,
            "chance_jaccard": chance,
            "units": {},
        }

        # ---- full-corpus models (saved for inspection) --------------------
        trained_full = run_trainings(
            [
                ("full_a", "a", None, algorithm, params),
                ("full_b", "b", None, algorithm, params),
            ],
            args.train_workers,
            f"full {algorithm}",
        )
        full_kv = {"a": trained_full["full_a"], "b": trained_full["full_b"]}
        for label in CORPORA:
            item(f"full {algorithm} corpus {label}: vocab {len(full_kv[label])}", indent=1)

        # ---- reliability via split-half RE-TRAIN --------------------------
        all_rel_rows: list[dict] = []
        for unit in units:
            targets = _be.build_targets(anchors, unit)
            names = sorted(targets)
            banner(f"Reliability unit={unit} ({len(names)} targets) algo={algorithm}")

            cos_samples: dict[str, dict[str, list[float]]] = {
                c: {n: [] for n in names} for c in CORPORA
            }
            jac_samples: dict[str, dict[str, list[float]]] = {
                c: {n: [] for n in names} for c in CORPORA
            }

            jobs = []
            for label in CORPORA:
                n_docs = len(documents[label])
                for split_i in range(args.n_splits):
                    order = rng.permutation(n_docs)
                    mid = n_docs // 2
                    # Distinct seeds per split so runs are not identical.
                    seed_l = int(cfg.seed + split_i * 2 + (0 if label == "a" else 1000))
                    seed_r = seed_l + 1
                    jobs.append((
                        (label, split_i, "l"), label, order[:mid],
                        algorithm, {**params, "seed": seed_l},
                    ))
                    jobs.append((
                        (label, split_i, "r"), label, order[mid:],
                        algorithm, {**params, "seed": seed_r},
                    ))

            trained = run_trainings(
                jobs, args.train_workers, f"{algorithm} split-half {unit}"
            )

            for label in CORPORA:
                for split_i in range(args.n_splits):
                    kv_l = trained.get((label, split_i, "l"))
                    kv_r = trained.get((label, split_i, "r"))
                    if kv_l is None or kv_r is None:
                        continue

                    for name in names:
                        terms = targets[name]
                        pl = second_order_profile(kv_l, terms, basis)
                        pr = second_order_profile(kv_r, terms, basis)
                        if np.linalg.norm(pl) == 0 or np.linalg.norm(pr) == 0:
                            continue
                        cos_samples[label][name].append(cosine(pl, pr))
                        exclude = set(terms)
                        nl = [w for w, _ in neighbors_from_kv(
                            kv_l, terms, k=top_k, exclude=exclude, restrict_to=basis
                        )]
                        nr = [w for w, _ in neighbors_from_kv(
                            kv_r, terms, k=top_k, exclude=exclude, restrict_to=basis
                        )]
                        if nl and nr:
                            jac_samples[label][name].append(jaccard_at_k(nl, nr))

            rows = []
            for name in names:
                record: dict = {
                    "unit": unit,
                    "target": name,
                    "backend": algorithm,
                }
                if unit == "anchor":
                    record["domain"] = anchors.by_term(name).domain
                    record["polysemous"] = anchors.by_term(name).polysemous
                else:
                    record["domain"] = name
                    record["polysemous"] = False
                for label in CORPORA:
                    f = count_freq(documents[label], targets[name])
                    cs = cos_samples[label][name]
                    js = jac_samples[label][name]
                    split_half = float(np.mean(cs)) if cs else 0.0
                    record[f"f_{label}"] = f
                    record[f"split_half_{label}"] = round(split_half, 4)
                    record[f"reliability_{label}"] = round(spearman_brown(split_half), 4)
                    record[f"neighbour_jaccard_{label}"] = (
                        round(float(np.mean(js)), 4) if js else 0.0
                    )
                record["min_reliability"] = round(
                    min(record["reliability_a"], record["reliability_b"]), 4
                )
                record["min_f"] = min(record["f_a"], record["f_b"])
                record["usable"] = bool(
                    record["min_reliability"] >= threshold
                    and record["min_f"] >= min_freq_floor
                )
                record["neighbours_above_chance"] = bool(
                    min(record["neighbour_jaccard_a"], record["neighbour_jaccard_b"])
                    > 2 * chance
                )
                rows.append(record)

            table = pd.DataFrame(rows).sort_values("min_reliability", ascending=False)
            all_rel_rows.extend(rows)
            save_table(
                table,
                out_path("tables", "p2", 13, "reliability", unit, algorithm),
                parquet=False,
            )
            item(f"{'target':<24}{'f_A':>7}{'f_B':>7}{'relA':>7}{'relB':>7}{'jacA':>7}{'jacB':>7}  usable")
            for _, row in table.head(12).iterrows():
                flag = "yes" if row["usable"] else "NO"
                item(
                    f"{row['target']:<24}{row['f_a']:>7}{row['f_b']:>7}"
                    f"{row['reliability_a']:>7.3f}{row['reliability_b']:>7.3f}"
                    f"{row['neighbour_jaccard_a']:>7.3f}{row['neighbour_jaccard_b']:>7.3f}  {flag}"
                )
            usable = int(table["usable"].sum())
            above = int(table["neighbours_above_chance"].sum())
            algo_summary["units"][unit] = {
                "n_targets": len(names),
                "n_usable": usable,
                "n_neighbours_above_chance": above,
                "mean_reliability_a": round(float(table["reliability_a"].mean()), 4),
                "mean_reliability_b": round(float(table["reliability_b"].mean()), 4),
                "mean_neighbour_jaccard_a": round(float(table["neighbour_jaccard_a"].mean()), 4),
                "mean_neighbour_jaccard_b": round(float(table["neighbour_jaccard_b"].mean()), 4),
            }
            item(
                f"{unit}: usable {usable}/{len(names)}; "
                f"neighbour Jaccard mean A={algo_summary['units'][unit]['mean_neighbour_jaccard_a']:.3f} "
                f"B={algo_summary['units'][unit]['mean_neighbour_jaccard_b']:.3f} "
                f"(chance {chance:.3f})"
            )

        save_summary(
            algo_summary,
            out_path("summaries", "p2", 13, "reliability", algorithm, ext="json"),
        )

        # reliability lookup for comparison
        reliability: dict[tuple[str, str], dict] = {}
        for row in all_rel_rows:
            reliability[(row["unit"], row["target"])] = {
                "a": row["reliability_a"],
                "b": row["reliability_b"],
                "usable": row["usable"],
            }

        # ---- full-corpus semantic comparison ------------------------------
        # Train bootstrap/permutation models ONCE; score every unit from them.
        compare_summary: dict = {
            "algorithm": algorithm,
            "basis_size": len(basis),
            "chance_jaccard": chance,
            "units": {},
        }
        neighbor_long: list[dict] = []
        unit_targets = {u: _be.build_targets(anchors, u) for u in units}

        banner(f"Bootstrap ({args.n_bootstrap}) + permutation ({args.n_permutations}) for {algorithm}")
        boot_cosines: dict[tuple[str, str], list[float]] = {}
        perm_cosines: dict[tuple[str, str], list[float]] = {}
        for unit, targets in unit_targets.items():
            for name in targets:
                boot_cosines[(unit, name)] = []
                perm_cosines[(unit, name)] = []

        boot_jobs = []
        for b_i in range(args.n_bootstrap):
            ia = rng.integers(0, len(documents["a"]), len(documents["a"]))
            ib = rng.integers(0, len(documents["b"]), len(documents["b"]))
            boot_jobs.append((
                ("boot", b_i, "a"), "a", ia,
                algorithm, {**params, "seed": int(cfg.seed + 10_000 + b_i)},
            ))
            boot_jobs.append((
                ("boot", b_i, "b"), "b", ib,
                algorithm, {**params, "seed": int(cfg.seed + 20_000 + b_i)},
            ))
        boot_trained = run_trainings(
            boot_jobs, args.train_workers, f"{algorithm} bootstrap"
        )
        for b_i in range(args.n_bootstrap):
            kv_a = boot_trained.get(("boot", b_i, "a"))
            kv_b = boot_trained.get(("boot", b_i, "b"))
            if kv_a is None or kv_b is None:
                continue
            for unit, targets in unit_targets.items():
                pa = profiles_for_targets(kv_a, targets, basis)
                pb = profiles_for_targets(kv_b, targets, basis)
                for name in targets:
                    if np.linalg.norm(pa[name]) == 0 or np.linalg.norm(pb[name]) == 0:
                        continue
                    boot_cosines[(unit, name)].append(cosine(pa[name], pb[name]))

        pooled = documents["a"] + documents["b"]
        n_a = len(documents["a"])
        try:
            wa = pd.read_parquet(
                analysis_dir / "corpus_a.parquet", columns=["word_count"]
            )["word_count"].to_numpy()
            wb = pd.read_parquet(
                analysis_dir / "corpus_b.parquet", columns=["word_count"]
            )["word_count"].to_numpy()
            lengths = np.concatenate([wa, wb])
        except Exception:
            lengths = np.array(
                [len(d) for d in documents["a"]] + [len(d) for d in documents["b"]],
                dtype=float,
            )
        deciles = pd.qcut(lengths, 10, labels=False, duplicates="drop")

        perm_jobs = []
        for p_i in range(args.n_permutations):
            mask = np.zeros(len(pooled), dtype=bool)
            for stratum in np.unique(deciles):
                idx = np.flatnonzero(deciles == stratum)
                k = int(round(len(idx) * n_a / len(pooled)))
                chosen = rng.choice(idx, size=min(k, len(idx)), replace=False)
                mask[chosen] = True
            perm_jobs.append((
                ("perm", p_i, "a"), "pooled", np.flatnonzero(mask),
                algorithm, {**params, "seed": int(cfg.seed + 30_000 + p_i)},
            ))
            perm_jobs.append((
                ("perm", p_i, "b"), "pooled", np.flatnonzero(~mask),
                algorithm, {**params, "seed": int(cfg.seed + 40_000 + p_i)},
            ))
        perm_trained = run_trainings(
            perm_jobs, args.train_workers, f"{algorithm} permutation"
        )
        for p_i in range(args.n_permutations):
            kv_pa = perm_trained.get(("perm", p_i, "a"))
            kv_pb = perm_trained.get(("perm", p_i, "b"))
            if kv_pa is None or kv_pb is None:
                continue
            for unit, targets in unit_targets.items():
                pa = profiles_for_targets(kv_pa, targets, basis)
                pb = profiles_for_targets(kv_pb, targets, basis)
                for name in targets:
                    if np.linalg.norm(pa[name]) == 0 or np.linalg.norm(pb[name]) == 0:
                        continue
                    perm_cosines[(unit, name)].append(cosine(pa[name], pb[name]))

        for unit in units:
            targets = unit_targets[unit]
            names = sorted(targets)
            banner(f"Compare unit={unit} algo={algorithm}")
            observed_a = profiles_for_targets(full_kv["a"], targets, basis)
            observed_b = profiles_for_targets(full_kv["b"], targets, basis)

            rows = []
            for name in names:
                va, vb = observed_a[name], observed_b[name]
                observed = cosine(va, vb) if (np.linalg.norm(va) and np.linalg.norm(vb)) else 0.0
                rel = reliability.get(
                    (unit, name), {"a": 0.0, "b": 0.0, "usable": False}
                )
                ceiling = reliability_ceiling(rel["a"], rel["b"])
                exclude = set(targets[name])
                na = [w for w, _ in neighbors_from_kv(
                    full_kv["a"], targets[name], k=top_k, exclude=exclude, restrict_to=basis
                )]
                nb = [w for w, _ in neighbors_from_kv(
                    full_kv["b"], targets[name], k=top_k, exclude=exclude, restrict_to=basis
                )]
                for corpus_label, neigh in (("a", na), ("b", nb)):
                    for rank, word in enumerate(neigh, start=1):
                        neighbor_long.append({
                            "backend": algorithm,
                            "unit": unit,
                            "target": name,
                            "corpus": corpus_label,
                            "rank": rank,
                            "neighbor": word,
                        })

                null = np.asarray(perm_cosines[(unit, name)], dtype=float)
                boot = np.asarray(boot_cosines[(unit, name)], dtype=float)
                p_value = (
                    float((np.sum(null <= observed) + 1) / (len(null) + 1))
                    if len(null) else np.nan
                )
                rows.append({
                    "unit": unit,
                    "target": name,
                    "backend": algorithm,
                    "domain": anchors.by_term(name).domain if unit == "anchor" else name,
                    "polysemous": anchors.by_term(name).polysemous if unit == "anchor" else False,
                    "f_a": count_freq(documents["a"], targets[name]),
                    "f_b": count_freq(documents["b"], targets[name]),
                    "reliability_a": round(rel["a"], 4),
                    "reliability_b": round(rel["b"], 4),
                    "reliability_ceiling": round(ceiling, 4),
                    "cosine_raw": round(observed, 4),
                    "cos_star": round(attenuation_correct(observed, rel["a"], rel["b"]), 4),
                    "rbo": round(rank_biased_overlap(na, nb, k=top_k), 4) if na and nb else 0.0,
                    "jaccard_at_k": round(jaccard_at_k(na, nb), 4) if na and nb else 0.0,
                    "jaccard_chance": round(chance, 4),
                    "perm_p": round(p_value, 5) if p_value == p_value else np.nan,
                    "null_mean": round(float(null.mean()), 4) if len(null) else np.nan,
                    "boot_ci_low": round(float(np.percentile(boot, 2.5)), 4) if len(boot) else np.nan,
                    "boot_ci_high": round(float(np.percentile(boot, 97.5)), 4) if len(boot) else np.nan,
                    "in_fdr_family": bool(rel["usable"]),
                    "neighbours_a": "; ".join(na),
                    "neighbours_b": "; ".join(nb),
                })

            table = pd.DataFrame(rows)
            family = table["in_fdr_family"].to_numpy()
            q = np.full(len(table), np.nan)
            if family.any() and table.loc[family, "perm_p"].notna().any():
                q[family] = benjamini_hochberg(
                    table.loc[family, "perm_p"].fillna(1.0).to_numpy()
                )
            table["fdr_q"] = np.round(q, 5)
            table["significant"] = table["fdr_q"] < fdr_q
            table = table.sort_values(["in_fdr_family", "cos_star"], ascending=[False, False])
            save_table(
                table,
                out_path("tables", "p2", 14, "semantic_comparison", unit, algorithm),
                parquet=False,
            )
            in_family = table[table["in_fdr_family"]]
            compare_summary["units"][unit] = {
                "n_targets": len(table),
                "n_in_family": int(len(in_family)),
                "n_significant": int(in_family["significant"].sum()) if len(in_family) else 0,
                "mean_cos_star": round(float(in_family["cos_star"].mean()), 4) if len(in_family) else None,
                "mean_jaccard": round(float(in_family["jaccard_at_k"].mean()), 4) if len(in_family) else None,
            }
            item(f"{'target':<20}{'cos*':>7}{'jac':>7}{'relA':>7}{'relB':>7}{'usable':>8}")
            for _, row in table.iterrows():
                if not row["in_fdr_family"]:
                    continue
                item(
                    f"{row['target']:<20}{row['cos_star']:>7.3f}{row['jaccard_at_k']:>7.3f}"
                    f"{row['reliability_a']:>7.3f}{row['reliability_b']:>7.3f}{'yes':>8}"
                )

        # save second-order full profiles as npz for figures
        for unit in units:
            targets = _be.build_targets(anchors, unit)
            names = sorted(targets)
            for label in CORPORA:
                matrix = np.vstack([
                    profiles_for_targets(full_kv[label], targets, basis)[n] for n in names
                ])
                destination = vectors_dir / f"{unit}_{label}_{algorithm}.npz"
                np.savez_compressed(
                    destination,
                    matrix=matrix,
                    names=np.array(names, dtype=object),
                    basis=np.array(basis, dtype=object),
                )
                item(f"wrote {destination.relative_to(ROOT)}")

        if neighbor_long:
            save_table(
                pd.DataFrame(neighbor_long),
                out_path("tables", "p2", 14, "neighbors_top15", algorithm),
                parquet=False,
            )
            item(f"neighbours long-form rows: {len(neighbor_long)}")

        save_summary(
            compare_summary,
            out_path("summaries", "p2", 14, "semantic_comparison", algorithm, ext="json"),
        )

        domain_stats = algo_summary["units"].get("domain", {})
        if domain_stats.get("n_usable", 0) == 0 and not args.no_gate:
            item(
                f"NOTE: {algorithm} domain reliability gate has 0 usable targets "
                f"— report as negative methodological result (expected at small N)."
            )

    banner("Done")
    item("Compare all methods with: python scripts/23_methods_compare.py")
    item("Word2Vec/FastText are evaluated empirically; low reliability is a result, not a skip.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
