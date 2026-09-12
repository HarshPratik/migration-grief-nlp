"""Phase 5 Task 4 - early vs late PPMI graphs and betweenness spikes.

New.docx: Year 1 vs Year 3+ co-occurrence (PPMI), NetworkX betweenness.
Nodes whose betweenness spikes later are candidate hybrid anchors
(points de capiton). Dual clocks: author-relative T0 and calendar time.

Extras: seed-domain overlay, permutation null for betweenness deltas,
Jaccard of top-betweenness sets, GraphML for Gephi.

  python scripts/43_temporal_graphs.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analysis_config import load_analysis_config, load_anchors, load_stopword_policy  # noqa: E402
from analysis_io import as_token_list, banner, item, make_rng, out_path, save_summary, save_table  # noqa: E402
from temporal import cfg_temporal  # noqa: E402
from vectors import build_cooccurrence, build_vocabulary, ppmi  # noqa: E402

CHUNK = 64


def _load(analysis_dir: Path, label: str) -> pd.DataFrame:
    path = analysis_dir / f"corpus_{label}_temporal.parquet"
    if not path.exists():
        raise SystemExit(f"missing {path}; run scripts/40_temporal_stratify.py first")
    df = pd.read_parquet(path)
    df["tokens_lemma"] = df["tokens_lemma"].map(as_token_list)
    return df


# Closed-class items that survive English stoplists and otherwise dominate
# late-period betweenness (calendar-late slices contain more non-English posts).
MULTILINGUAL_FUNCTION = frozenset(
    """
    el la los las un una uno de del al que en y a por para con sin
    es son fue fueron ser estar hay muy mas más como pero si no ya lo le
    ich du er sie es wir ihr mein dein nicht und der die das den dem des
    ein eine einem einen auf mit von zu im ist sind war auch nur
    le la les un une des du au aux et je tu il elle nous vous
    est sont pas pour dans sur avec sans qui moi toi
    il lo la gli le un uno una di del della dei delle che e per
    non si sono è una
    ang mga ng sa na ko ako ikaw siya hindi at o kung
    que qué porque como esta este esto esa ese eso una
    una mas más vez fue son para como
    """.split()
)


def _units(df: pd.DataFrame, policy, *, min_len: int = 4, protect: frozenset[str] | None = None) -> list[list[str]]:
    """Lemma units with stopwords and short tokens removed so betweenness is not
    dominated by function words (especially other-language closed-class items).
    Seed-domain terms are always kept.
    """
    keep = protect or frozenset()
    units: list[list[str]] = []
    for tokens in df["tokens_lemma"]:
        cleaned = []
        for raw in tokens:
            word = str(raw).lower()
            if word in keep:
                cleaned.append(word)
                continue
            if word in MULTILINGUAL_FUNCTION:
                continue
            if len(word) < min_len or not word.isalpha():
                continue
            if policy.is_stopword(word):
                continue
            cleaned.append(word)
        units.append(cleaned)
    return units


def _ppmi_graph(
    units: list[list[str]],
    *,
    window: int,
    min_count: int,
    min_edge_count: int,
    min_ppmi: float,
    top_edges: int,
) -> tuple[nx.Graph, pd.DataFrame, pd.DataFrame]:
    vocab = build_vocabulary(units, min_count=min_count)
    if len(vocab) < 3 or not units:
        empty = nx.Graph()
        return empty, pd.DataFrame(), pd.DataFrame()
    index = {w: i for i, w in enumerate(vocab)}
    acc = sp.csr_matrix((len(vocab), len(vocab)), dtype=np.float64)
    for start in range(0, len(units), CHUNK):
        acc = acc + build_cooccurrence(units[start: start + CHUNK], vocab, window=window, symmetric=True)
    weighted = ppmi(acc)
    coo = weighted.tocoo()
    rows = []
    for i, j, w in zip(coo.row, coo.col, coo.data):
        if i >= j:
            continue
        raw = acc[i, j]
        if raw < min_edge_count or w < min_ppmi:
            continue
        rows.append({"source": vocab[i], "target": vocab[j], "ppmi": float(w), "count": float(raw)})
    edges = pd.DataFrame(rows)
    if edges.empty:
        return nx.Graph(), edges, pd.DataFrame()
    edges = edges.sort_values(["ppmi", "source", "target"], ascending=[False, True, True]).head(int(top_edges))
    graph = nx.Graph()
    for _, row in edges.iterrows():
        graph.add_edge(row["source"], row["target"], weight=float(row["ppmi"]), count=float(row["count"]))
    between = nx.betweenness_centrality(graph, weight="weight", normalized=True)
    nodes = pd.DataFrame(
        {"node": list(between), "betweenness": list(between.values()), "degree": [graph.degree(n) for n in between]}
    ).sort_values("betweenness", ascending=False)
    return graph, edges, nodes


def _domain_of(term: str, by_domain: dict[str, set[str]]) -> str | None:
    for name, vocab in by_domain.items():
        if term in vocab:
            return name
    return None


def _compare_slices(
    early_nodes: pd.DataFrame,
    late_nodes: pd.DataFrame,
    *,
    spike_ratio: float,
    by_domain: dict[str, set[str]],
    rng: np.random.Generator,
    n_perm: int,
) -> pd.DataFrame:
    early = dict(zip(early_nodes["node"], early_nodes["betweenness"]))
    late = dict(zip(late_nodes["node"], late_nodes["betweenness"]))
    nodes = sorted(set(early) | set(late))
    rows = []
    deltas = []
    for node in nodes:
        b0 = float(early.get(node, 0.0))
        b1 = float(late.get(node, 0.0))
        ratio = b1 / (b0 + 1e-12)
        delta = b1 - b0
        deltas.append(delta)
        rows.append({
            "node": node,
            "betweenness_early": b0,
            "betweenness_late": b1,
            "delta": delta,
            "ratio": ratio,
            "spike": bool(ratio >= spike_ratio and b1 > 0),
            "domain": _domain_of(node, by_domain) or "",
        })
    table = pd.DataFrame(rows)
    if table.empty or n_perm < 1:
        table["perm_p"] = np.nan
        return table
    observed = table["delta"].to_numpy()
    null_extreme = np.zeros(len(observed), dtype=int)
    # Permute the paired early/late labels by shuffling deltas around a 0-centered null:
    # swap early/late for a random half of nodes each permutation.
    for _ in range(int(n_perm)):
        sign = rng.choice(np.array([-1.0, 1.0]), size=len(observed))
        perm = observed * sign
        null_extreme += (np.abs(perm) >= np.abs(observed)).astype(int)
    table["perm_p"] = (null_extreme + 1) / (int(n_perm) + 1)
    return table


def main() -> int:
    parser = argparse.ArgumentParser(description="Early vs late PPMI graphs and betweenness spikes")
    args = parser.parse_args()
    cfg = load_analysis_config()
    anchors = load_anchors()
    knobs = cfg_temporal(cfg)
    rng = make_rng(cfg.seed)
    analysis_dir = cfg.path("analysis_dir")
    window = int(cfg.get("cooccurrence.window"))
    min_count = int(cfg.get("cooccurrence.min_count"))
    min_edge = int(cfg.get("cooccurrence.min_edge_count"))
    min_ppmi = float(cfg.get("cooccurrence.min_ppmi"))
    top_edges = int(cfg.get("cooccurrence.top_edges"))
    by_domain = {name: {a.term for a in domain.anchors} for name, domain in anchors.domains.items()}
    protect = frozenset(term for vocab in by_domain.values() for term in vocab)
    policy = load_stopword_policy()
    networks_dir = cfg.path("outputs_dir") / "networks"

    banner("Temporal graph topology")
    item("Year 1 vs Year 3+ PPMI; betweenness spike = hybrid candidate")

    node_rows = []
    compare_rows = []
    summary = {}
    for label in ("a", "b"):
        frame = _load(analysis_dir, label)
        item(f"corpus {label}  {len(frame):,} posts")
        for clock, early_col, late_col in (
            ("t0", "slice_year1", "slice_year3plus"),
            ("calendar", "cal_year1", "cal_year3plus"),
        ):
            early_df = frame[frame[early_col]]
            late_df = frame[frame[late_col]]
            item(f"  {clock}: year1={len(early_df):,}  year3+={len(late_df):,}")
            graphs = {}
            for slice_name, slice_df in (("year1", early_df), ("year3plus", late_df)):
                graph, edges, nodes = _ppmi_graph(
                    _units(slice_df, policy, protect=protect),
                    window=window, min_count=min_count, min_edge_count=min_edge,
                    min_ppmi=min_ppmi, top_edges=top_edges,
                )
                graphs[slice_name] = (graph, edges, nodes)
                if not edges.empty:
                    save_table(
                        edges,
                        out_path("tables", "p5", 43, "edges", label, clock, slice_name),
                    )
                    save_table(
                        nodes,
                        out_path("tables", "p5", 43, "nodes", label, clock, slice_name),
                    )
                    graphml = networks_dir / f"p5_43_{label}_{clock}_{slice_name}.graphml"
                    graphml.parent.mkdir(parents=True, exist_ok=True)
                    nx.write_graphml(graph, graphml)
                for _, row in nodes.iterrows():
                    node_rows.append({
                        "corpus": label, "clock": clock, "slice": slice_name,
                        "node": row["node"], "betweenness": row["betweenness"],
                        "degree": row["degree"],
                        "domain": _domain_of(str(row["node"]), by_domain) or "",
                    })
            early_nodes = graphs["year1"][2]
            late_nodes = graphs["year3plus"][2]
            if early_nodes.empty or late_nodes.empty:
                item(f"  {clock}: skipped comparison (empty slice graph)")
                summary[f"{label}_{clock}"] = {
                    "n_early": int(len(early_df)), "n_late": int(len(late_df)),
                    "empty_graph": True,
                }
                continue
            compared = _compare_slices(
                early_nodes, late_nodes,
                spike_ratio=knobs["betweenness_spike_ratio"],
                by_domain=by_domain, rng=rng, n_perm=knobs["n_permutations"],
            )
            compared["corpus"] = label
            compared["clock"] = clock
            top_e = set(early_nodes.head(25)["node"])
            top_l = set(late_nodes.head(25)["node"])
            jaccard = len(top_e & top_l) / max(len(top_e | top_l), 1)
            compared["top25_jaccard"] = jaccard
            # Hybrid: betweenness spike AND the node is a seed-domain term, or
            # neighbours in the late graph include two different seed domains.
            late_graph = graphs["year3plus"][0]
            hybrid_nodes = []
            for _, row in compared.iterrows():
                if not row["spike"]:
                    continue
                node = str(row["node"])
                neighbor_domains = {
                    _domain_of(nb, by_domain)
                    for nb in late_graph.neighbors(node)
                    if _domain_of(nb, by_domain)
                }
                own = _domain_of(node, by_domain)
                if own is not None or len(neighbor_domains) >= 2:
                    hybrid_nodes.append(node)
            compared["hybrid"] = compared["node"].isin(hybrid_nodes)
            compare_rows.append(compared)
            save_table(compared, out_path("tables", "p5", 43, "betweenness_delta", label, clock))
            summary[f"{label}_{clock}"] = {
                "n_early": int(len(early_df)),
                "n_late": int(len(late_df)),
                "n_nodes_compared": int(len(compared)),
                "n_spikes": int(compared["spike"].sum()),
                "n_hybrids": int(compared["hybrid"].sum()),
                "top25_jaccard": float(jaccard),
                "empty_graph": False,
            }

    if compare_rows:
        save_table(pd.concat(compare_rows, ignore_index=True), out_path("tables", "p5", 43, "betweenness_delta"))
    if node_rows:
        save_table(pd.DataFrame(node_rows), out_path("tables", "p5", 43, "betweenness_nodes"))
    save_summary(
        {
            "task": "temporal_graph_topology",
            "year1": f"months < {knobs['year1_max_months']}",
            "year3plus": f"months >= {knobs['year3_min_months']}",
            "spike_ratio": knobs["betweenness_spike_ratio"],
            "slices": summary,
        },
        out_path("summaries", "p5", 43, "graphs", ext="json"),
    )
    banner("Done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
