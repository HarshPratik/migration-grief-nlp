"""Phase 3 - PPMI-weighted co-occurrence network, exported for Gephi.

Counts how often two words appear together, reweights the raw counts with PPMI
so that "grief + counsellor" outranks "grief + get", and writes the surviving
pairs as an edge list, a node table and a GraphML file.

Three scopes, because "together" is not one thing:

  window   +/-N tokens inside one post (default). The narrowest, most
           syntagmatic reading of association; uses `tokens_lemma`.
  sentence any two words in the same sentence, from `tokens_ctx` grouped by
           `sent_idx`. NOTE `tokens_ctx` keeps stopwords - it feeds the emotion
           models, which need their negators - so this scope filters them out
           through config/stopwords.yaml before counting. It also works on
           SURFACE forms, not lemmas, so its vocabulary is not identical to the
           other two scopes' and the two edge lists are not row-comparable.
  doc      any two words in the same post, counted ONCE per post. Repeats are
           collapsed deliberately: without that, a single 1,700-token post
           would contribute ~1.4M pair events and dominate every edge weight.

Counting never crosses a unit boundary - `vectors.build_cooccurrence` scans each
unit independently - and the unit is a post (window/doc) or a sentence
(sentence). Nothing links the last token of one post to the first of the next.

Co-occurrence is chunked over units and the per-chunk sparse matrices are
summed. Same arithmetic as one call, bounded memory: doc scope on Corpus A
generates ~4.2M pair events, which a single call would materialise as ~1GB of
Python int lists.

Examples:
    D:\\research_paper\\.venv-analysis\\Scripts\\python.exe scripts/16_cooccurrence.py
    ... scripts/16_cooccurrence.py --corpus a --scope sentence
    ... scripts/16_cooccurrence.py --scope doc --min-ppmi 1.0 --top-edges 500
    ... scripts/16_cooccurrence.py --anchors-only --min-count 2
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analysis_config import load_analysis_config, load_anchors, load_stopword_policy  # noqa: E402
from analysis_io import as_token_list, banner, item, out_path, save_summary, save_table  # noqa: E402
from vectors import build_cooccurrence, build_vocabulary, ppmi  # noqa: E402

# Units per sparse-matrix accumulation step. Purely a memory knob: the summed
# result is identical for any value, so it is not a config threshold.
CHUNK_UNITS = 64


# --------------------------------------------------------------------------
# units
# --------------------------------------------------------------------------


def window_units(df: pd.DataFrame) -> list[list[str]]:
    """One unit per post, lemmas in reading order (window and doc scope)."""
    return [as_token_list(tokens) for tokens in df["tokens_lemma"]]


def sentence_units(df: pd.DataFrame, policy, *, min_len: int, anchor_terms: frozenset[str]) -> list[list[str]]:
    """One unit per sentence, from `tokens_ctx` grouped by `sent_idx`.

    `tokens_ctx` retains stopwords by design, so the stoplist is applied here.
    An anchor always survives filtering - it is the unit of analysis, and a
    stoplist must never be able to delete one (same rule as scripts/08).
    """
    units: list[list[str]] = []
    for tokens_raw, sents_raw in zip(df["tokens_ctx"], df["sent_idx"]):
        tokens = as_token_list(tokens_raw)
        sents = as_token_list(sents_raw)
        if len(tokens) != len(sents):
            raise SystemExit(
                f"tokens_ctx ({len(tokens)}) and sent_idx ({len(sents)}) are not aligned; "
                "re-run scripts/08_reprocess_tokens.py"
            )
        grouped: dict[str, list[str]] = {}
        for token, sentence in zip(tokens, sents):
            word = token.lower()
            keep = word in anchor_terms or (
                len(word) >= min_len and word.isalpha() and not policy.is_stopword(word)
            )
            if keep:
                grouped.setdefault(sentence, []).append(word)
        units.extend(grouped.values())
    return units


def collapse_to_sets(units: list[list[str]]) -> list[list[str]]:
    """Unique tokens per unit, sorted. Turns doc scope into co-document counts."""
    return [sorted(set(unit)) for unit in units]


def build_units(df: pd.DataFrame, scope: str, policy, *, min_len: int, anchor_terms: frozenset[str]) -> list[list[str]]:
    if scope in ("window", "doc"):
        units = window_units(df)
        return collapse_to_sets(units) if scope == "doc" else units
    if scope == "sentence":
        return sentence_units(df, policy, min_len=min_len, anchor_terms=anchor_terms)
    raise SystemExit(f"unknown scope {scope!r}")


def effective_window(scope: str, units: list[list[str]], window: int) -> int:
    """Window scope uses the configured span; the others span the whole unit."""
    if scope == "window":
        return max(int(window), 1)
    return max((len(unit) for unit in units), default=1) or 1


# --------------------------------------------------------------------------
# counting
# --------------------------------------------------------------------------


def accumulate_cooccurrence(
    units: list[list[str]],
    vocabulary: tuple[str, ...],
    *,
    window: int,
    chunk: int = CHUNK_UNITS,
) -> sp.csr_matrix:
    """Sum `build_cooccurrence` over chunks of units. Bounded memory, same result."""
    size = len(vocabulary)
    total = sp.csr_matrix((size, size), dtype=np.float64)
    for start in range(0, len(units), chunk):
        total = total + build_cooccurrence(
            units[start : start + chunk], vocabulary, window=window, symmetric=True
        )
    return sp.csr_matrix(total)


def edge_frame(
    counts: sp.csr_matrix,
    weights: sp.csr_matrix,
    vocabulary: tuple[str, ...],
    *,
    min_ppmi: float,
    min_edge_count: int,
) -> pd.DataFrame:
    """Upper-triangle edge list, deterministic order: -ppmi, -count, source, target."""
    upper = sp.triu(weights, k=1).tocoo()
    candidate_rows = upper.row
    candidate_cols = upper.col
    raw = (
        np.asarray(counts[candidate_rows, candidate_cols]).ravel()
        if candidate_rows.size else np.zeros(0)
    )
    keep = (upper.data >= min_ppmi) & (raw >= min_edge_count)
    rows = candidate_rows[keep]
    cols = candidate_cols[keep]
    values = upper.data[keep]
    raw = raw[keep]
    frame = pd.DataFrame({
        "source": [vocabulary[i] for i in rows],
        "target": [vocabulary[j] for j in cols],
        "count": raw.astype(np.int64),
        "ppmi": values.astype(np.float64),
    })
    return frame.sort_values(
        ["ppmi", "count", "source", "target"], ascending=[False, False, True, True]
    ).reset_index(drop=True)


def node_frame(
    vocabulary: tuple[str, ...],
    frequency: Counter,
    edges: pd.DataFrame,
    domain_of: dict[str, str],
) -> pd.DataFrame:
    """One row per vocabulary type. `domain` is empty for a non-anchor."""
    degree: Counter = Counter()
    strength: dict[str, float] = {}
    for source, target, weight in zip(edges["source"], edges["target"], edges["ppmi"]):
        degree[source] += 1
        degree[target] += 1
        strength[source] = strength.get(source, 0.0) + float(weight)
        strength[target] = strength.get(target, 0.0) + float(weight)

    return pd.DataFrame({
        "node": list(vocabulary),
        "freq": [int(frequency.get(word, 0)) for word in vocabulary],
        "domain": [domain_of.get(word, "") for word in vocabulary],
        "degree": [int(degree.get(word, 0)) for word in vocabulary],
        "strength_ppmi": [float(strength.get(word, 0.0)) for word in vocabulary],
    }).sort_values(["freq", "node"], ascending=[False, True]).reset_index(drop=True)


def build_graph(edges: pd.DataFrame, nodes: pd.DataFrame, domain_of: dict[str, str]):
    """networkx graph over exactly the nodes touched by `edges`."""
    import networkx as nx

    freq = dict(zip(nodes["node"], nodes["freq"]))
    graph = nx.Graph()
    for source, target, count, weight in edges.itertuples(index=False):
        for word in (source, target):
            if word not in graph:
                graph.add_node(
                    word,
                    freq=int(freq.get(word, 0)),
                    domain=domain_of.get(word, ""),
                    is_anchor=bool(word in domain_of),
                )
        graph.add_edge(source, target, weight=float(weight), ppmi=float(weight), count=int(count))
    return graph


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def main() -> int:
    cfg = load_analysis_config()
    defaults = cfg.section("cooccurrence")

    parser = argparse.ArgumentParser(description="PPMI co-occurrence network per corpus")
    parser.add_argument("--corpus", choices=["a", "b", "both"], default="both")
    parser.add_argument("--scope", choices=["window", "sentence", "doc"], default=str(defaults["scope"]))
    parser.add_argument("--window", type=int, default=int(defaults["window"]))
    parser.add_argument("--min-count", type=int, default=int(defaults["min_count"]),
                        help="vocabulary floor: drop types occurring fewer times")
    parser.add_argument("--min-edge-count", type=int, default=int(defaults["min_edge_count"]),
                        help="edge support floor: drop pairs observed fewer times")
    parser.add_argument("--min-ppmi", type=float, default=float(defaults["min_ppmi"]),
                        help="edge association floor: drop pairs below this PPMI")
    parser.add_argument("--top-edges", type=int, default=int(defaults["top_edges"]),
                        help="how many edges enter the exported graph and the top-edge table")
    parser.add_argument("--anchors-only", action="store_true", default=bool(defaults["anchors_only"]),
                        help="restrict the vocabulary to seed-dictionary anchors")
    parser.add_argument("--analysis-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=None, help="recorded in provenance; no sampling here")
    args = parser.parse_args()

    if args.window < 1:
        parser.error("--window must be >= 1")
    if args.min_count < 1:
        parser.error("--min-count must be >= 1")
    if args.min_edge_count < 1:
        parser.error("--min-edge-count must be >= 1")
    if args.min_ppmi < 0 or not np.isfinite(args.min_ppmi):
        parser.error("--min-ppmi must be a finite value >= 0")
    if args.top_edges < 1:
        parser.error("--top-edges must be >= 1")

    cfg = load_analysis_config(seed=args.seed)
    anchors = load_anchors()
    policy = load_stopword_policy()

    analysis_dir = args.analysis_dir or cfg.path("analysis_dir")
    min_len = int(cfg.get("preprocess.min_token_len", 2))
    shift_k = int(cfg.get("vectors.ppmi.shift_k", 1))
    cds = float(cfg.get("vectors.ppmi.cds_smoothing", 1.0))

    anchor_terms = frozenset(a.term for a in anchors.anchors)
    domain_of = {a.term: a.domain for a in anchors.anchors}
    labels = ["a", "b"] if args.corpus == "both" else [args.corpus]

    banner("Settings")
    item(f"scope        {args.scope}" + (f" (+/-{args.window} tokens)" if args.scope == "window" else ""))
    item(f"vocabulary   min_count >= {args.min_count}" + ("  ANCHORS ONLY" if args.anchors_only else ""))
    item(f"edges        count >= {args.min_edge_count}, ppmi >= {args.min_ppmi}, top {args.top_edges} exported")
    item(f"ppmi         shift_k={shift_k}, cds_smoothing={cds}")
    if args.scope == "sentence":
        item(f"stopwords    {len(policy.effective)} filtered out of tokens_ctx (surface forms, not lemmas)")
    if args.scope == "doc":
        item("doc scope    each pair counted once per post, not once per token pair")

    summary: dict = {
        "scope": args.scope,
        "window": args.window if args.scope == "window" else None,
        "min_count": args.min_count,
        "min_edge_count": args.min_edge_count,
        "min_ppmi": args.min_ppmi,
        "top_edges": args.top_edges,
        "anchors_only": bool(args.anchors_only),
        "frequency_definition": "document frequency" if args.scope == "doc" else "token frequency",
        "ppmi": {"shift_k": shift_k, "cds_smoothing": cds},
        "seed": cfg.seed,
        "corpora": {},
    }

    for label in labels:
        source_path = analysis_dir / f"corpus_{label}.parquet"
        if not source_path.exists():
            raise SystemExit(f"missing {source_path}; run scripts/08_reprocess_tokens.py first")

        columns = ["tokens_lemma"] if args.scope in ("window", "doc") else ["tokens_ctx", "sent_idx"]
        df = pd.read_parquet(source_path, columns=columns)

        banner(f"Corpus {label.upper()} - {args.scope} co-occurrence")
        units = build_units(df, args.scope, policy, min_len=min_len, anchor_terms=anchor_terms)
        frequency: Counter = Counter()
        for unit in units:
            frequency.update(unit)

        vocabulary = build_vocabulary(units, min_count=args.min_count)
        if args.anchors_only:
            vocabulary = tuple(word for word in vocabulary if word in anchor_terms)
        if len(vocabulary) < 2:
            raise SystemExit(
                f"corpus {label}: vocabulary is {len(vocabulary)} type(s) at min_count="
                f"{args.min_count}{' with --anchors-only' if args.anchors_only else ''}; nothing to link"
            )

        span = effective_window(args.scope, units, args.window)
        item(f"units        {len(units)} ({'posts' if args.scope != 'sentence' else 'sentences'})")
        item(f"tokens       {sum(frequency.values())}, vocabulary {len(vocabulary)} of {len(frequency)} types")
        item(f"span         {span} token(s)")

        counts = accumulate_cooccurrence(units, vocabulary, window=span)
        weights = ppmi(counts, shift_k=shift_k, cds_smoothing=cds, positive=True)
        edges = edge_frame(
            counts,
            weights,
            vocabulary,
            min_ppmi=args.min_ppmi,
            min_edge_count=args.min_edge_count,
        )

        pairs_possible = int(sp.triu(counts, k=1).nnz)
        item(f"pairs        {pairs_possible} co-occurring, {len(edges)} above ppmi {args.min_ppmi}")
        if edges.empty:
            raise SystemExit(
                f"corpus {label}: no edge clears ppmi >= {args.min_ppmi}. Lower --min-ppmi."
            )

        top = edges.head(args.top_edges).copy()
        nodes = node_frame(vocabulary, frequency, top, domain_of)
        graph = build_graph(top, nodes, domain_of)

        # ---- outputs ------------------------------------------------------
        written: list[str] = []
        edge_store = analysis_dir / f"cooccur_{label}_{args.scope}"
        written += [str(p) for p in save_table(edges, edge_store, parquet=True, csv=False)]
        written += [str(p) for p in save_table(
            top, out_path("tables", "p3", 16, f"edges_top{args.top_edges}", label), parquet=False
        )]
        written += [str(p) for p in save_table(
            nodes, out_path("tables", "p3", 16, "nodes", label), parquet=False
        )]

        import networkx as nx

        graph_path = out_path("networks", "p3", 16, "cooccur", label, ext="graphml")
        nx.write_graphml(graph, graph_path)
        written.append(str(graph_path))

        item(f"graph        {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges "
             f"(top {args.top_edges} by ppmi)")

        # ---- anchor-only subgraph per domain -------------------------------
        anchor_edges = edges[
            edges["source"].isin(anchor_terms) & edges["target"].isin(anchor_terms)
        ]
        domain_stats: dict[str, dict] = {}
        banner(f"Corpus {label.upper()} - anchor subgraphs by domain")
        for name, domain in anchors.domains.items():
            terms = set(domain.terms)
            subset = anchor_edges[
                anchor_edges["source"].isin(terms) & anchor_edges["target"].isin(terms)
            ].reset_index(drop=True)
            present = [t for t in sorted(terms) if t in frequency]
            in_vocab = [t for t in present if t in set(vocabulary)]
            subgraph = build_graph(subset, nodes, domain_of) if not subset.empty else None
            if subgraph is not None:
                sub_path = out_path("networks", "p3", 16, "cooccur", label, name, ext="graphml")
                nx.write_graphml(subgraph, sub_path)
                written.append(str(sub_path))
            domain_stats[name] = {
                "anchors_in_corpus": len(present),
                "anchors_in_vocabulary": len(in_vocab),
                "edges": int(len(subset)),
                "nodes": int(subgraph.number_of_nodes()) if subgraph is not None else 0,
            }
            item(
                f"{name:<12} {len(in_vocab):>3}/{len(domain.terms)} anchors in vocab, "
                f"{len(subset):>4} internal edge(s)"
                + ("" if subgraph is not None else "  - no subgraph written")
            )

        cross_domain = int(len(anchor_edges) - sum(d["edges"] for d in domain_stats.values()))
        item(f"cross-domain anchor edges: {cross_domain}")

        summary["corpora"][label] = {
            "n_units": len(units),
            "n_tokens": int(sum(frequency.values())),
            "n_types": len(frequency),
            "vocabulary_size": len(vocabulary),
            "span": span,
            "pairs_cooccurring": pairs_possible,
            "n_edges_above_min_ppmi": int(len(edges)),
            "n_edges_exported": int(graph.number_of_edges()),
            "n_nodes_exported": int(graph.number_of_nodes()),
            "ppmi_max": float(edges["ppmi"].max()),
            "ppmi_median": float(edges["ppmi"].median()),
            "edge_count_median": float(edges["count"].median()),
            "n_anchor_edges": int(len(anchor_edges)),
            "n_cross_domain_anchor_edges": cross_domain,
            "domains": domain_stats,
            "outputs": written,
        }

        singleton_share = float((top["count"] <= 1).mean())
        summary["corpora"][label]["singleton_edge_share_exported"] = round(singleton_share, 4)
        summary["corpora"][label]["node_metrics_scope"] = "exported top-edge graph"

    save_summary(summary, out_path("summaries", "p3", 16, "cooccurrence", ext="json"))
    banner("Done")
    for label in labels:
        stats = summary["corpora"][label]
        item(f"corpus {label}: {stats['n_nodes_exported']} nodes / {stats['n_edges_exported']} edges exported")
    item(f"summary -> {out_path('summaries', 'p3', 16, 'cooccurrence', ext='json').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
