"""Phase 4 brief: independent topic models + alignment + topic networks.

Fits LDA and NMF on Corpus A and Corpus B *separately* (k=10,12,15), then
optionally BERTopic if sentence-transformers is installed. CPU jobs run in
parallel; BERTopic embeddings use the GPU if CUDA is available.

  python scripts/33_phase4_full_compare.py
  python scripts/33_phase4_full_compare.py --skip-bertopic --workers 8
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analysis_config import load_analysis_config, load_anchors  # noqa: E402
from analysis_io import as_token_list, banner, item, out_path, save_summary, save_table  # noqa: E402
from vectors import codocument_counts, umass_coherence  # noqa: E402

K_GRID = (10, 12, 15)
TOP_N = 10
PROMINENT = 3


def _load_corpus(analysis_dir: Path, label: str) -> pd.DataFrame:
    path = analysis_dir / f"corpus_{label}.parquet"
    if not path.exists():
        raise SystemExit(f"missing {path}; run scripts/08_reprocess_tokens.py first")
    df = pd.read_parquet(path, columns=["post_id", "corpus", "tokens_lemma"])
    tokens = [as_token_list(t) for t in df["tokens_lemma"]]
    return pd.DataFrame({
        "post_id": df["post_id"].astype(str),
        "corpus": label,
        "tokens": tokens,
        "doc": [" ".join(t) for t in tokens],
    })


def _anchor_sets():
    cfg = load_anchors()
    by_domain: dict[str, set[str]] = {}
    for name, domain in cfg.domains.items():
        by_domain[name] = {a.term for a in domain.anchors}
    return by_domain


def _fit_one(payload: dict) -> dict:
    """Worker: fit one (method, corpus, k) model. Import sklearn inside."""
    from sklearn.decomposition import NMF, LatentDirichletAllocation
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

    docs: list[str] = payload["docs"]
    method = payload["method"]
    k = int(payload["k"])
    seed = int(payload["seed"])
    max_iter = int(payload["max_iter"])
    max_features = int(payload["max_features"])
    min_df = int(payload["min_df"])
    max_df = float(payload["max_df"])
    n_jobs = int(payload["n_jobs"])
    init = payload["init"]

    kwargs = dict(max_features=max_features, min_df=min_df, max_df=max_df, lowercase=False)
    if method == "nmf":
        vec = TfidfVectorizer(**kwargs)
        matrix = vec.fit_transform(docs)
        model = NMF(
            n_components=k, init=init, random_state=seed, max_iter=max_iter,
            solver="cd", beta_loss="frobenius",
        )
        theta = model.fit_transform(matrix)
        components = model.components_
    else:
        vec = CountVectorizer(**kwargs)
        matrix = vec.fit_transform(docs)
        model = LatentDirichletAllocation(
            n_components=k, random_state=seed, max_iter=max_iter,
            learning_method="online", learning_offset=50.0, batch_size=256,
            n_jobs=max(1, n_jobs), evaluate_every=-1,
        )
        theta = model.fit_transform(matrix)
        components = model.components_

    vocab = list(vec.get_feature_names_out())
    totals = theta.sum(axis=1, keepdims=True)
    theta_n = theta / np.where(totals > 0, totals, 1.0)
    mean_theta = theta_n.mean(axis=0)
    order = np.argsort(-components, axis=1)
    term_rows = []
    topics = []
    for t in range(k):
        idxs = order[t, :TOP_N]
        terms = []
        for rank, i in enumerate(idxs, start=1):
            w = float(components[t, i])
            term = vocab[i]
            terms.append((term, w))
            term_rows.append({
                "method": method, "corpus": payload["corpus"], "k": k,
                "topic": t, "rank": rank, "term": term, "weight": w,
            })
        topics.append({
            "topic": t,
            "mean_theta": float(mean_theta[t]),
            "top_terms": [x[0] for x in terms],
            "weights": [x[1] for x in terms],
        })
    return {
        "method": method,
        "corpus": payload["corpus"],
        "k": k,
        "n_docs": len(docs),
        "vocab_size": len(vocab),
        "topics": topics,
        "term_rows": term_rows,
        "mean_theta": mean_theta.tolist(),
        "topic_term": components.tolist(),
        "vocabulary": vocab,
        "theta": theta_n.tolist(),
    }


def _align_topic(top_terms: list[str], by_domain: dict[str, set[str]]) -> dict:
    counts = {d: sum(1 for w in top_terms if w in terms) for d, terms in by_domain.items()}
    hits = {d: [w for w in top_terms if w in terms] for d, terms in by_domain.items()}
    return {
        **{f"n_{d}": counts[d] for d in by_domain},
        **{f"terms_{d}": ",".join(hits[d]) for d in by_domain},
        "mix_digital_absence": bool(counts.get("digital", 0) and counts.get("absence", 0)),
        "mix_digital_temporal": bool(counts.get("digital", 0) and counts.get("temporal", 0)),
        "mix_identity_temporal": bool(counts.get("identity", 0) and counts.get("temporal", 0)),
    }


def _coherence(topics: list[dict], token_lists: list[list[str]]) -> float:
    all_terms = [w for t in topics for w in t["top_terms"]]
    dfreq, codf = codocument_counts(token_lists, all_terms)
    scores = [umass_coherence(t["top_terms"], dfreq, codf) for t in topics]
    return float(np.mean(scores)) if scores else 0.0


def _write_html(result: dict, path: Path) -> None:
    """Lightweight interactive topic view (pyLDAvis-style, no extra dep)."""
    from sklearn.decomposition import PCA

    tt = np.asarray(result["topic_term"], dtype=np.float64)
    row = tt / np.where(tt.sum(axis=1, keepdims=True) > 0, tt.sum(axis=1, keepdims=True), 1.0)
    xy = PCA(n_components=2, random_state=0).fit_transform(row)
    sizes = np.asarray(result["mean_theta"], dtype=np.float64)
    payload = []
    for i, topic in enumerate(result["topics"]):
        payload.append({
            "id": i,
            "x": float(xy[i, 0]),
            "y": float(xy[i, 1]),
            "size": float(sizes[i]),
            "terms": [{"word": w, "weight": wt} for w, wt in zip(topic["top_terms"], topic["weights"])],
        })
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"/><title>LDA/NMF {result['method']} corpus {result['corpus']} k={result['k']}</title>
<style>
body{{font:16px/1.4 Georgia,serif;margin:24px;background:#f7f6f3;color:#222}}
svg{{background:#fff;border:1px solid #ddd}}
.panel{{margin-top:12px}}
circle{{fill:#1f4e79;fill-opacity:.7;stroke:#123;cursor:pointer}}
circle.sel{{fill:#e45756}}
#terms{{min-height:220px}}
</style></head><body>
<h1>{result['method'].upper()} · corpus {result['corpus'].upper()} · k={result['k']}</h1>
<p>Click a topic. Circle size is mean document share. Layout is PCA of topic-term weights.</p>
<svg id="viz" width="720" height="420"></svg>
<div class="panel" id="terms">Click a circle.</div>
<script>
const T = {json.dumps(payload)};
const svg = document.getElementById('viz');
const xs = T.map(t=>t.x), ys = T.map(t=>t.y), ss = T.map(t=>t.size);
const xmin=Math.min(...xs), xmax=Math.max(...xs), ymin=Math.min(...ys), ymax=Math.max(...ys);
const smin=Math.min(...ss), smax=Math.max(...ss);
function nx(x){{return 40+640*((x-xmin)/Math.max(xmax-xmin,1e-9));}}
function ny(y){{return 40+340*((y-ymin)/Math.max(ymax-ymin,1e-9));}}
function nr(s){{return 10+28*((s-smin)/Math.max(smax-smin,1e-9));}}
function show(t){{
  document.querySelectorAll('circle').forEach(c=>c.classList.remove('sel'));
  document.getElementById('c'+t.id).classList.add('sel');
  document.getElementById('terms').innerHTML = '<h2>Topic '+t.id+' (share '+t.size.toFixed(3)+')</h2><ol>'+
    t.terms.map(w=>'<li>'+w.word+' <small>'+w.weight.toFixed(3)+'</small></li>').join('')+'</ol>';
}}
T.forEach(t=>{{
  const c=document.createElementNS('http://www.w3.org/2000/svg','circle');
  c.setAttribute('id','c'+t.id);
  c.setAttribute('cx', nx(t.x)); c.setAttribute('cy', ny(t.y)); c.setAttribute('r', nr(t.size));
  c.addEventListener('click', ()=>show(t));
  svg.appendChild(c);
  const lab=document.createElementNS('http://www.w3.org/2000/svg','text');
  lab.setAttribute('x', nx(t.x)+nr(t.size)+4); lab.setAttribute('y', ny(t.y)+4);
  lab.setAttribute('font-size','12'); lab.textContent='T'+t.id;
  svg.appendChild(lab);
}});
</script></body></html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")


def _topic_network(token_lists: list[list[str]], terms: set[str], by_domain: dict[str, set[str]], window: int = 5):
    import networkx as nx

    pair = Counter()
    freq = Counter()
    wanted = {t for t in terms if t}
    for tokens in token_lists:
        idx = [tok for tok in tokens if tok in wanted]
        freq.update(idx)
        for i, w in enumerate(tokens):
            if w not in wanted:
                continue
            lo, hi = max(0, i - window), min(len(tokens), i + window + 1)
            for j in range(lo, hi):
                u, v = w, tokens[j]
                if u == v or v not in wanted:
                    continue
                a, b = (u, v) if u < v else (v, u)
                pair[(a, b)] += 1
    g = nx.Graph()
    domain_of = {}
    for d, s in by_domain.items():
        for t in s:
            domain_of[t] = d
    for node, f in freq.items():
        g.add_node(node, freq=int(f), domain=domain_of.get(node, "other"))
    for (a, b), c in pair.items():
        if c >= 3:
            g.add_edge(a, b, weight=float(c), count=int(c))
    bridges = []
    for u, v, data in g.edges(data=True):
        du, dv = g.nodes[u].get("domain"), g.nodes[v].get("domain")
        if du != dv and du != "other" and dv != "other":
            bridges.append({"u": u, "v": v, "du": du, "dv": dv, "count": data["count"]})
    return g, bridges


def _run_minilm(docs: list[str], corpus: str, k: int, seed: int) -> dict:
    """MiniLM embeddings + KMeans + class-based TF-IDF (BERTopic-like, no umap/hdbscan)."""
    import torch
    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.feature_extraction.text import TfidfVectorizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    emb = embedder.encode(
        docs, batch_size=256, show_progress_bar=False,
        convert_to_numpy=True, normalize_embeddings=True,
    )
    km = MiniBatchKMeans(n_clusters=k, random_state=seed, batch_size=2048, n_init=10)
    labels = km.fit_predict(emb)
    cluster_docs, sizes = [], []
    for c in range(k):
        members = [docs[i] for i, lab in enumerate(labels) if int(lab) == c]
        sizes.append(len(members))
        cluster_docs.append(" ".join(members) if members else "")
    vec = TfidfVectorizer(max_features=5000, min_df=1, max_df=0.95)
    matrix = vec.fit_transform(cluster_docs)
    vocab = list(vec.get_feature_names_out())
    components = matrix.toarray()
    n = max(len(docs), 1)
    mean_theta = np.asarray(sizes, dtype=np.float64) / n
    order = np.argsort(-components, axis=1)
    rows, topic_payload = [], []
    for t in range(k):
        terms = []
        for rank, i in enumerate(order[t, :TOP_N], start=1):
            term, wt = vocab[i], float(components[t, i])
            terms.append((term, wt))
            rows.append({
                "method": "minilm", "corpus": corpus, "k": k,
                "topic": t, "rank": rank, "term": term, "weight": wt,
            })
        topic_payload.append({
            "topic": t, "mean_theta": float(mean_theta[t]),
            "top_terms": [w for w, _ in terms], "weights": [w for _, w in terms],
        })
    return {
        "method": "minilm",
        "corpus": corpus,
        "k": k,
        "n_docs": len(docs),
        "device": device,
        "vocab_size": len(vocab),
        "topics": topic_payload,
        "term_rows": rows,
        "mean_theta": mean_theta.tolist(),
        "topic_term": components.tolist(),
        "vocabulary": vocab,
        "outlier_frac": 0.0,
    }


def _run_bertopic(docs: list[str], corpus: str, k: int, seed: int) -> dict | None:
    try:
        from bertopic import BERTopic
        from sentence_transformers import SentenceTransformer
    except Exception:
        return _run_minilm(docs, corpus, k, seed)

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    model = BERTopic(
        embedding_model=embedder,
        nr_topics=k,
        calculate_probabilities=False,
        verbose=False,
    )
    topics, _ = model.fit_transform(docs)
    info = model.get_topic_info()
    rows = []
    topic_payload = []
    for _, rec in info.iterrows():
        tid = int(rec["Topic"])
        if tid < 0:
            continue
        words = model.get_topic(tid) or []
        top = [(str(w), float(wt)) for w, wt in words[:TOP_N]]
        topic_payload.append({
            "topic": tid,
            "mean_theta": float(rec.get("Count", 0)) / max(len(docs), 1),
            "top_terms": [w for w, _ in top],
            "weights": [wt for _, wt in top],
        })
        for rank, (w, wt) in enumerate(top, start=1):
            rows.append({
                "method": "bertopic", "corpus": corpus, "k": k,
                "topic": tid, "rank": rank, "term": w, "weight": wt,
            })
    return {
        "method": "bertopic",
        "corpus": corpus,
        "k": k,
        "n_docs": len(docs),
        "device": device,
        "topics": topic_payload,
        "term_rows": rows,
        "outlier_frac": float(np.mean(np.array(topics) < 0)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 4 independent topic comparison")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lda-jobs", type=int, default=4)
    parser.add_argument("--skip-bertopic", action="store_true")
    parser.add_argument("--skip-sklearn", action="store_true")
    parser.add_argument("--k", default="10,12,15")
    parser.add_argument("--max-iter-lda", type=int, default=20)
    parser.add_argument("--max-iter-nmf", type=int, default=200)
    args = parser.parse_args()
    ks = tuple(int(x) for x in str(args.k).split(",") if x.strip())

    cfg = load_analysis_config()
    by_domain = _anchor_sets()
    analysis_dir = cfg.path("analysis_dir")
    corpora = {lab: _load_corpus(analysis_dir, lab) for lab in ("a", "b")}
    seed = int(cfg.get("seed"))
    max_features = int(cfg.get("topics.max_features"))
    min_df = int(cfg.get("topics.min_df"))
    max_df = float(cfg.get("topics.max_df"))
    init = str(cfg.get("topics.nmf_init"))

    jobs = []
    if not args.skip_sklearn:
        for lab, frame in corpora.items():
            docs = frame["doc"].tolist()
            for method, max_iter in (("lda", args.max_iter_lda), ("nmf", args.max_iter_nmf)):
                for k in ks:
                    jobs.append({
                        "method": method, "corpus": lab, "k": k, "docs": docs,
                        "seed": seed, "max_iter": max_iter, "max_features": max_features,
                        "min_df": min_df, "max_df": max_df, "n_jobs": args.lda_jobs,
                        "init": init,
                    })

    banner("Phase 4 independent models")
    item(f"{len(jobs)} sklearn jobs (LDA+NMF × A/B × k={list(ks)}) workers={args.workers}")

    results: list[dict] = []
    if jobs:
        with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futs = {pool.submit(_fit_one, job): job for job in jobs}
            for fut in as_completed(futs):
                job = futs[fut]
                try:
                    res = fut.result()
                except Exception as exc:
                    item(f"FAILED {job['method']} {job['corpus']} k={job['k']}: {exc}")
                    continue
                results.append(res)
                item(f"ok {res['method']} corpus={res['corpus']} k={res['k']} vocab={res['vocab_size']}")

    if not args.skip_bertopic:
        for lab, frame in corpora.items():
            item(f"MiniLM/BERTopic corpus={lab} on GPU if available")
            b = _run_bertopic(frame["doc"].tolist(), lab, k=12, seed=seed)
            if b:
                results.append(b)
                if b.get("error"):
                    item(f"BERTopic skip: {b['error']}")
                else:
                    item(
                        f"ok {b['method']} corpus={lab} device={b.get('device')} "
                        f"outlier={b.get('outlier_frac'):.3f}"
                    )

    term_rows = [r for res in results for r in res.get("term_rows") or []]
    if term_rows:
        terms_path = out_path("tables", "p4", 33, "topic_terms")
        terms_df = pd.DataFrame(term_rows)
        if args.skip_sklearn and terms_path.exists():
            prev = pd.read_csv(terms_path)
            drop = set(terms_df["method"].unique())
            prev = prev[~prev["method"].isin(drop)]
            terms_df = pd.concat([prev, terms_df], ignore_index=True)
        save_table(terms_df, terms_path, parquet=False)

    align_rows = []
    for res in results:
        if not res.get("topics"):
            continue
        tokens = corpora[res["corpus"]]["tokens"].tolist()
        coh = _coherence(res["topics"], tokens) if res["method"] != "bertopic" else None
        res["umass"] = coh
        for t in res["topics"]:
            al = _align_topic(t["top_terms"], by_domain)
            align_rows.append({
                "method": res["method"], "corpus": res["corpus"], "k": res["k"],
                "topic": t["topic"], "mean_theta": t["mean_theta"],
                "top_terms": " ".join(t["top_terms"]),
                "umass_model": coh,
                **al,
            })
    align_df = pd.DataFrame(align_rows)
    if not align_df.empty:
        align_path = out_path("tables", "p4", 33, "alignment")
        if args.skip_sklearn and align_path.exists():
            prev = pd.read_csv(align_path)
            drop = set(align_df["method"].unique())
            prev = prev[~prev["method"].isin(drop)]
            align_df = pd.concat([prev, align_df], ignore_index=True)
        save_table(align_df, align_path, parquet=False)

    mix = (
        align_df[align_df["k"] == 12]
        .groupby(["method", "corpus"], as_index=False)["mix_digital_absence"]
        .sum()
        if not align_df.empty else pd.DataFrame()
    )
    if not mix.empty:
        save_table(mix, out_path("tables", "p4", 33, "digital_absence_mix"), parquet=False)

    # Interactive HTML for primary LDA k=12
    for res in results:
        if res.get("method") in ("lda", "nmf", "minilm") and res.get("k") == 12 and "topic_term" in res:
            html_path = out_path("figures", "p4", 33, "interactive", res["method"], res["corpus"], ext="html")
            _write_html(res, html_path)
            docs_copy = ROOT / "docs" / "figures" / html_path.name
            docs_copy.parent.mkdir(parents=True, exist_ok=True)
            docs_copy.write_text(html_path.read_text(encoding="utf-8"), encoding="utf-8")
            item(f"interactive {html_path.name}")

    # Topic-keyword networks for LDA k=12 prominent topics
    try:
        import networkx as nx
        from plotting import apply_style, fig_network, save_fig
        style = apply_style(cfg.section("figures"))
        colors = {
            "temporal": "#4C78A8", "identity": "#72B7B2",
            "absence": "#E45756", "digital": "#F58518", "other": "#BAB0AC",
        }
        all_bridges = []
        for res in results:
            if res.get("method") != "lda" or res.get("k") != 12:
                continue
            lab = res["corpus"]
            ranked = sorted(res["topics"], key=lambda t: -t["mean_theta"])[:PROMINENT]
            terms = set()
            for t in ranked:
                terms.update(t["top_terms"])
            for d, s in by_domain.items():
                if d in ("identity", "temporal", "digital", "absence"):
                    terms.update(s)
            g, bridges = _topic_network(corpora[lab]["tokens"].tolist(), terms, by_domain)
            gpath = out_path("networks", "p4", 33, "topic_cooccur", lab, ext="graphml")
            gpath.parent.mkdir(parents=True, exist_ok=True)
            nx.write_graphml(g, gpath)
            node_colors = {n: colors.get(g.nodes[n].get("domain"), colors["other"]) for n in g.nodes}
            sizes = {n: 40 + 18 * np.log1p(float(g.nodes[n].get("freq", 1))) for n in g.nodes}
            if g.number_of_nodes():
                fig, _ = fig_network(
                    g, style=style,
                    title=f"Corpus {lab.upper()} LDA k=12 topic-keyword network",
                    node_colors=node_colors, node_sizes=sizes, label_top_n=28,
                )
                pngs = save_fig(fig, out_path("figures", "p4", 33, "topic_network", lab),
                                formats=list(cfg.get("figures.formats")))
                for p in pngs:
                    dest = ROOT / "docs" / "figures" / p.name
                    dest.write_bytes(p.read_bytes())
            for b in bridges:
                all_bridges.append({"corpus": lab, **b})
        if all_bridges:
            save_table(pd.DataFrame(all_bridges), out_path("tables", "p4", 33, "domain_bridges"), parquet=False)
    except Exception as exc:
        item(f"network figures skipped: {exc}")

    summary = {
        "brief": "Independent LDA/NMF (and BERTopic if available) on each corpus, k in 10-15",
        "n_models": (
            int(align_df.groupby(["method", "corpus", "k"]).ngroups)
            if not align_df.empty else len(results)
        ),
        "k_grid": list(ks),
        "digital_absence_mix_k12": mix.to_dict(orient="records") if not mix.empty else [],
        "models": [
            {
                "method": r.get("method"), "corpus": r.get("corpus"), "k": r.get("k"),
                "n_docs": r.get("n_docs"), "umass": r.get("umass"),
                "error": r.get("error"), "device": r.get("device"),
            }
            for r in results
        ],
    }
    save_summary(summary, out_path("summaries", "p4", 33, "full_compare", ext="json"))
    item("wrote outputs/summaries/p4_33_full_compare.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
