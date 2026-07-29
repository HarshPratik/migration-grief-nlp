"""Phase 2 - GATE 1. Tier every anchor and every domain by measured coverage.

This runs before any modelling, and it is a gate rather than a report: if a
domain has too little data to analyse, the run stops here instead of quietly
emitting an empty table three scripts later.

Tiering uses the MINIMUM across corpora of BOTH token frequency and document
frequency:
  * minimum across corpora, because a cross-corpus comparison is limited by its
    weaker side - `funeral` at 0/241 cannot support a similarity measure no
    matter how frequent it is in Corpus B;
  * document frequency as well as token frequency, because 12 tokens spread
    over 10 posts is 10 independent observations, not 12.

Anchors that fail the floor are NOT dropped from the study. They are routed to
keyness, prevalence and concordance (script 10), which stay valid at low counts.
An anchor present in one corpus and absent from the other is a finding, not a
failure - it is exactly what a keyness test is designed to detect.

Example:
    D:\\research_paper\\.venv-analysis\\Scripts\\python.exe scripts/09_anchor_coverage.py
    ... scripts/09_anchor_coverage.py --no-gate      # report without exiting non-zero
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analysis_config import load_analysis_config, load_anchors  # noqa: E402
from analysis_io import banner, item, out_path, save_summary, save_table  # noqa: E402
from plotting import apply_style, fig_grouped_bar, save_fig  # noqa: E402

TIER_ANALYZABLE = 1
TIER_CAVEATED = 2
TIER_EXCLUDED = 3

TIER_TREATMENT = {
    1: "full inference (cos*, permutation p, bootstrap CI); enters FDR family",
    2: "descriptive only (effect size + CI, no p-values); excluded from FDR",
    3: "keyness / prevalence / concordance only",
}


def corpus_counts(path: Path, column: str = "tokens_lemma") -> tuple[Counter, Counter, int]:
    """Token frequency, document frequency, and post count for one corpus."""
    df = pd.read_parquet(path)
    freq: Counter = Counter()
    docs: Counter = Counter()
    for tokens in df[column]:
        tokens = list(tokens)
        freq.update(tokens)
        docs.update(set(tokens))
    return freq, docs, len(df)


def assign_tier(f_a: int, df_a: int, f_b: int, df_b: int, tiers: dict) -> int:
    min_f = min(f_a, f_b)
    min_df = min(df_a, df_b)
    t1, t2 = tiers["tier1"], tiers["tier2"]
    if min_f >= t1["min_freq"] and min_df >= t1["min_docfreq"]:
        return TIER_ANALYZABLE
    if min_f >= t2["min_freq"] and min_df >= t2["min_docfreq"]:
        return TIER_CAVEATED
    return TIER_EXCLUDED


def main() -> int:
    parser = argparse.ArgumentParser(description="Tier anchors and domains by measured coverage")
    parser.add_argument("--analysis-dir", type=Path, default=None)
    parser.add_argument("--column", default="tokens_lemma")
    parser.add_argument(
        "--no-gate", action="store_true",
        help="report only; do not exit non-zero when a domain is unusable",
    )
    args = parser.parse_args()

    cfg = load_analysis_config()
    anchors = load_anchors()
    tiers = {k: dict(v) for k, v in cfg.section("tiers").items() if isinstance(v, type(cfg.section("tiers")))}
    tiers = {"tier1": dict(cfg.get("tiers.tier1")), "tier2": dict(cfg.get("tiers.tier2"))}
    analysis_dir = args.analysis_dir or cfg.path("analysis_dir")

    counts: dict[str, tuple[Counter, Counter, int]] = {}
    for label in ("a", "b"):
        path = analysis_dir / f"corpus_{label}.parquet"
        if not path.exists():
            raise SystemExit(f"missing {path}; run scripts/08_reprocess_tokens.py first")
        counts[label] = corpus_counts(path, args.column)

    freq_a, docs_a, n_a = counts["a"]
    freq_b, docs_b, n_b = counts["b"]

    banner("Corpus sizes")
    item(f"A: {n_a} posts, {sum(freq_a.values())} tokens, vocab {len(freq_a)}")
    item(f"B: {n_b} posts, {sum(freq_b.values())} tokens, vocab {len(freq_b)}")
    item(f"tier1 floor: min_freq >= {tiers['tier1']['min_freq']}, min_docfreq >= {tiers['tier1']['min_docfreq']}")
    item(f"tier2 floor: min_freq >= {tiers['tier2']['min_freq']}, min_docfreq >= {tiers['tier2']['min_docfreq']}")

    # ---- per-anchor -------------------------------------------------------
    rows = []
    for anchor in anchors.anchors:
        f_a, d_a = freq_a.get(anchor.term, 0), docs_a.get(anchor.term, 0)
        f_b, d_b = freq_b.get(anchor.term, 0), docs_b.get(anchor.term, 0)
        tier = assign_tier(f_a, d_a, f_b, d_b, tiers)
        rows.append({
            "domain": anchor.domain,
            "anchor": anchor.term,
            "multiword": anchor.is_multiword,
            "tier_expected": anchor.tier,
            "f_a": f_a, "df_a": d_a, "pct_posts_a": round(100 * d_a / max(n_a, 1), 2),
            "f_b": f_b, "df_b": d_b, "pct_posts_b": round(100 * d_b / max(n_b, 1), 2),
            "min_f": min(f_a, f_b), "min_df": min(d_a, d_b),
            "tier_observed": tier,
            "treatment": TIER_TREATMENT[tier],
            "corpus_specific": (f_a == 0) != (f_b == 0),
        })
    anchor_table = pd.DataFrame(rows).sort_values(
        ["tier_observed", "domain", "min_f"], ascending=[True, True, False]
    )
    save_table(anchor_table, out_path("tables", "p2", 9, "anchor_coverage"), parquet=False)

    # ---- per-domain (the primary unit of analysis) ------------------------
    domain_rows = []
    for name, domain in anchors.domains.items():
        terms = domain.terms
        # `strict` excludes anchors flagged polysemous in anchors.yaml. This
        # matters: the Digital domain's raw total is dominated by `connection`
        # and `disconnect`, which concordance shows are overwhelmingly
        # interpersonal ("a close connection"), not technological. Reporting
        # only the full total would overstate what the domain measures.
        strict_terms = tuple(a.term for a in domain.anchors if not a.polysemous)
        f_a = sum(freq_a.get(t, 0) for t in terms)
        f_b = sum(freq_b.get(t, 0) for t in terms)
        sf_a = sum(freq_a.get(t, 0) for t in strict_terms)
        sf_b = sum(freq_b.get(t, 0) for t in strict_terms)
        # Document frequency at domain level = posts containing ANY of its
        # anchors, so it cannot be summed from per-anchor df.
        d_a = _domain_docfreq(analysis_dir / "corpus_a.parquet", terms, args.column)
        d_b = _domain_docfreq(analysis_dir / "corpus_b.parquet", terms, args.column)
        sd_a = _domain_docfreq(analysis_dir / "corpus_a.parquet", strict_terms, args.column)
        sd_b = _domain_docfreq(analysis_dir / "corpus_b.parquet", strict_terms, args.column)
        tier = assign_tier(f_a, d_a, f_b, d_b, tiers)
        strict_tier = assign_tier(sf_a, sd_a, sf_b, sd_b, tiers)
        domain_rows.append({
            "domain": name,
            "label": domain.label,
            "n_anchors": len(terms),
            "n_polysemous": len(terms) - len(strict_terms),
            "f_a": f_a, "df_a": d_a, "pct_posts_a": round(100 * d_a / max(n_a, 1), 2),
            "f_b": f_b, "df_b": d_b, "pct_posts_b": round(100 * d_b / max(n_b, 1), 2),
            "strict_f_a": sf_a, "strict_df_a": sd_a,
            "strict_pct_posts_a": round(100 * sd_a / max(n_a, 1), 2),
            "strict_f_b": sf_b, "strict_df_b": sd_b,
            "strict_pct_posts_b": round(100 * sd_b / max(n_b, 1), 2),
            "min_f": min(f_a, f_b), "min_df": min(d_a, d_b),
            "tier_observed": tier,
            "tier_strict": strict_tier,
            "usable": tier <= TIER_CAVEATED,
            "usable_strict": strict_tier <= TIER_CAVEATED,
        })
    domain_table = pd.DataFrame(domain_rows)
    save_table(domain_table, out_path("tables", "p2", 9, "domain_coverage"), parquet=False)

    # ---- report -----------------------------------------------------------
    banner("Anchor tiers")
    tier_counts = anchor_table["tier_observed"].value_counts().to_dict()
    for tier in (1, 2, 3):
        item(f"tier {tier}: {tier_counts.get(tier, 0):>3} anchors - {TIER_TREATMENT[tier]}")
    item(f"corpus-specific (present in one corpus only): {int(anchor_table['corpus_specific'].sum())}")

    banner("Usable anchors by domain")
    item(f"{'domain':<12}{'tier1':>7}{'tier2':>7}{'tier3':>7}{'usable':>8}")
    for name in anchors.domains:
        subset = anchor_table[anchor_table["domain"] == name]
        counts_by_tier = subset["tier_observed"].value_counts()
        usable = int((subset["tier_observed"] <= TIER_CAVEATED).sum())
        item(
            f"{name:<12}{counts_by_tier.get(1, 0):>7}{counts_by_tier.get(2, 0):>7}"
            f"{counts_by_tier.get(3, 0):>7}{usable:>4}/{len(subset)}"
        )

    banner("Domain-level coverage (the primary unit)")
    item("'strict' drops anchors flagged polysemous in anchors.yaml")
    item(f"{'domain':<12}{'f_A':>7}{'%postsA':>9}{'f_B':>8}{'%postsB':>9}{'tier':>6}"
         f"{'strictA':>9}{'strictB':>9}{'sTier':>7}")
    for row in domain_rows:
        item(
            f"{row['domain']:<12}{row['f_a']:>7}{row['pct_posts_a']:>8.1f}%"
            f"{row['f_b']:>8}{row['pct_posts_b']:>8.1f}%{row['tier_observed']:>6}"
            f"{row['strict_f_a']:>9}{row['strict_f_b']:>9}{row['tier_strict']:>7}"
        )
    downgraded = [
        (r["domain"], r["tier_observed"], r["tier_strict"])
        for r in domain_rows if r["tier_strict"] > r["tier_observed"]
    ]
    if downgraded:
        item("WARNING - tier depends on polysemous anchors:")
        for name, full, strict in downgraded:
            item(f"{name}: tier {full} on all anchors, tier {strict} on unambiguous ones only", indent=2)
        item("Report the strict figures alongside the full ones; the full total", indent=2)
        item("does not measure only what the domain name claims.", indent=2)

    banner("Tier 1 and 2 anchors")
    usable_rows = anchor_table[anchor_table["tier_observed"] <= TIER_CAVEATED]
    if usable_rows.empty:
        item("none")
    for _, row in usable_rows.iterrows():
        item(
            f"T{row['tier_observed']} {row['domain']:<10} {row['anchor']:<22}"
            f"A={row['f_a']:>5}/{row['df_a']:<4} B={row['f_b']:>5}/{row['df_b']:<4}"
        )

    # ---- figure -----------------------------------------------------------
    style = apply_style(cfg.section("figures"))
    domain_names = [r["domain"] for r in domain_rows]
    fig, _ = fig_grouped_bar(
        domain_names,
        {
            cfg.get("figures.corpus_labels.a"): [r["pct_posts_a"] for r in domain_rows],
            cfg.get("figures.corpus_labels.b"): [r["pct_posts_b"] for r in domain_rows],
        },
        style=style,
        title="Seed-dictionary coverage: posts containing any anchor",
        ylabel="% of posts",
        colors={
            cfg.get("figures.corpus_labels.a"): cfg.get("figures.corpus_colors.a"),
            cfg.get("figures.corpus_labels.b"): cfg.get("figures.corpus_colors.b"),
        },
    )
    written = save_fig(fig, out_path("figures", "p2", 9, "domain_coverage", ext="png"),
                       formats=list(cfg.get("figures.formats")))

    unusable = [r["domain"] for r in domain_rows if not r["usable"]]
    summary = {
        "corpus_sizes": {
            "a": {"posts": n_a, "tokens": sum(freq_a.values()), "vocab": len(freq_a)},
            "b": {"posts": n_b, "tokens": sum(freq_b.values()), "vocab": len(freq_b)},
        },
        "tiers": tiers,
        "anchor_tier_counts": {str(k): int(v) for k, v in tier_counts.items()},
        "n_anchors": len(anchors.anchors),
        "n_usable_anchors": int(len(usable_rows)),
        "domains": domain_rows,
        "unusable_domains": unusable,
        "figures": [str(p) for p in written],
    }
    save_summary(summary, out_path("summaries", "p2", 9, "anchor_coverage", ext="json"))

    banner("Gate")
    if unusable:
        item(f"UNUSABLE DOMAIN(S): {unusable}")
        item("These cannot support embedding, topic or anchor-level emotion analysis.")
        item("They remain valid for prevalence and exhaustive concordance (script 10).")
        if not args.no_gate:
            item("Exiting non-zero. Re-run with --no-gate to continue anyway.")
            return 1
    else:
        item("all domains usable")
    return 0


def _domain_docfreq(path: Path, terms, column: str) -> int:
    """Posts containing at least one anchor from the domain."""
    df = pd.read_parquet(path, columns=[column])
    wanted = set(terms)
    return int(sum(1 for tokens in df[column] if wanted & set(tokens)))


if __name__ == "__main__":
    raise SystemExit(main())
