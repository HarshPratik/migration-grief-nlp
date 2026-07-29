"""Phase 2 - re-tokenise with a corrected stopword policy and anchor collapsing.

Reads data/interim (the filtered, pre-NLP tables) and writes data/analysis.
NEVER writes to data/processed: the Phase 1 snapshot stays frozen.

Four corrections to scripts/04_preprocess_nlp.py, each of which changes results:

1. STOPWORDS. Phase 1 used spaCy's `tok.is_stop`, whose 326-word list deletes
   `alone`, `empty`, `gone`(->"go"), `name` and every negator. That silently
   emptied the author's Absence dictionary. Replaced with the explicit,
   version-controlled list in config/stopwords.yaml.
2. NO PRE-LOWERCASING. Phase 1 lowercased before spaCy, which degrades tagging
   and lemmatisation of proper nouns and sentence-initial words.
3. DIGITS KEPT. Phase 1's NON_ALPHA_RE stripped all digits, destroying
   "2 years", "3 hours behind" and every date.
4. SENTENCE BOUNDARIES. Phase 1 disabled both parser and senter, so no sentence
   index existed. Stage 4 needs one to clip context windows.

Two token columns are emitted because they answer different questions:
  tokens_lemma - stopword-filtered lemmas. Feeds vectors, co-occurrence, topics.
  tokens_ctx   - surface tokens with STOPWORDS RETAINED, plus a parallel
                 sent_idx. Feeds windows, emotion and ABSA, because "+/-5 words"
                 means surface words and any emotion model needs its negators.

Multi-word anchors are collapsed to a single token before either column is
built - `video call` and `culture shock` can never clear a frequency floor as
separate tokens.

Example:
    D:\\research_paper\\.venv-analysis\\Scripts\\python.exe scripts/08_reprocess_tokens.py
    ... scripts/08_reprocess_tokens.py --corpus a --limit 50
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analysis_config import load_analysis_config, load_anchors, load_stopword_policy  # noqa: E402
from analysis_io import banner, item, save_summary, save_table  # noqa: E402
from text_windows import SurfaceMatcher  # noqa: E402

INTERIM_DIR = ROOT / "data" / "interim"

# Cleaning applied BEFORE spaCy. Deliberately conservative: it removes markup
# and platform noise but preserves case, digits and sentence punctuation.
FOOTER_RE = re.compile(r"(?:&#32;\s*)*submitted by.*?(?:\[link\]|\[comments\]).*$", re.I | re.S)
HTML_TAG_RE = re.compile(r"<[^>]+>")
ENTITY_RE = re.compile(r"&#\d+;|&[a-z]+;", re.I)
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
USER_RE = re.compile(r"/?u/[\w-]+", re.I)
SUBREDDIT_RE = re.compile(r"/?r/[\w-]+", re.I)
MARKDOWN_RE = re.compile(r"[*_~`#>\[\]()]+")
WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Strip markup and platform noise. Case, digits and punctuation survive."""
    if not isinstance(text, str) or not text:
        return ""
    out = FOOTER_RE.sub("", text)
    out = HTML_TAG_RE.sub(" ", out)
    out = ENTITY_RE.sub(" ", out)
    out = URL_RE.sub(" ", out)
    out = USER_RE.sub(" ", out)
    out = SUBREDDIT_RE.sub(" ", out)
    out = MARKDOWN_RE.sub(" ", out)
    return WHITESPACE_RE.sub(" ", out).strip()


def load_pipeline():
    """en_core_web_sm with senter enabled and the heavy components off."""
    import spacy

    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
    # `pipe_names` lists only ENABLED components; senter ships disabled, so
    # check `component_names` before trying to add a duplicate.
    if "senter" in nlp.component_names:
        nlp.enable_pipe("senter")
    else:
        nlp.add_pipe("senter")
    return nlp


def process_document(doc, matcher: SurfaceMatcher, policy, *, min_len: int, anchor_terms: frozenset[str]):
    """Turn one spaCy Doc into the two aligned token views.

    Returns (tokens_ctx, sent_idx, tokens_lemma, n_anchor_hits).
    """
    surfaces = [tok.text for tok in doc]
    lemmas = [tok.lemma_ for tok in doc]
    keep = [(not tok.is_space) and (not tok.is_punct) for tok in doc]

    sentence_of = [0] * len(doc)
    for index, sent in enumerate(doc.sents):
        for tok in sent:
            sentence_of[tok.i] = index

    tokens_ctx: list[str] = []
    sent_idx: list[int] = []
    tokens_lemma: list[str] = []
    hits = 0

    for emitted, start, span in matcher.collapse(surfaces):
        is_anchor = emitted in anchor_terms
        if is_anchor:
            hits += 1
            tokens_ctx.append(emitted)
            sent_idx.append(sentence_of[start])
            # An anchor always survives into the lemma view: it is the unit of
            # analysis, so a stoplist must never be able to remove it.
            tokens_lemma.append(emitted)
            continue

        if span != 1 or not keep[start]:
            continue

        surface = surfaces[start].lower()
        if not any(ch.isalnum() for ch in surface):
            continue

        tokens_ctx.append(surface)
        sent_idx.append(sentence_of[start])

        lemma = lemmas[start].lower().strip()
        if (
            len(lemma) >= min_len
            and lemma.isalpha()
            and not policy.is_stopword(lemma)
            and not policy.is_stopword(surface)
        ):
            tokens_lemma.append(lemma)

    return tokens_ctx, sent_idx, tokens_lemma, hits


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Re-tokenise with corrected stopwords, digits, sentences and anchor collapsing"
    )
    parser.add_argument("--corpus", choices=["a", "b", "both"], default="both")
    parser.add_argument("--input-dir", type=Path, default=INTERIM_DIR)
    parser.add_argument("--limit", type=int, default=0, help="process only the first N posts (debug)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = load_analysis_config(seed=args.seed)
    anchors = load_anchors()
    policy = load_stopword_policy()

    matcher = SurfaceMatcher.from_surface_map(anchors.surface_to_term)
    anchor_terms = frozenset(a.term for a in anchors.anchors)
    min_len = int(cfg.get("preprocess.min_token_len", 2))
    analysis_dir = cfg.path("analysis_dir")
    analysis_dir.mkdir(parents=True, exist_ok=True)

    banner("Stopword policy")
    item(f"source            {policy.source}")
    item(f"protected         {len(policy.protect)} words ({len(policy.protect & set(w for w in policy.protect))} listed)")
    item(f"effective list    {len(policy.effective)} words")
    item(f"anchors collapsed {len(anchors.anchors)} terms, {len(matcher.multi)} multi-token forms")

    nlp = load_pipeline()
    labels = ["a", "b"] if args.corpus == "both" else [args.corpus]
    summary: dict = {"corpora": {}, "stopwords": {
        "source": policy.source,
        "n_protected": len(policy.protect),
        "n_effective": len(policy.effective),
        "protected": sorted(policy.protect),
    }}

    for label in labels:
        source = args.input_dir / f"corpus_{label}.parquet"
        if not source.exists():
            raise SystemExit(f"missing {source}; run scripts/03_filter_and_normalize.py first")

        df = pd.read_parquet(source)
        if args.limit:
            df = df.head(args.limit)

        banner(f"Preprocess corpus {label.upper()}")
        item(f"input {len(df)} posts from {source.relative_to(ROOT)}")

        texts = [clean_text(t) for t in df["text"].fillna("").astype(str)]

        ctx_all: list[list[str]] = []
        sent_all: list[list[int]] = []
        lemma_all: list[list[str]] = []
        anchor_hits: list[int] = []

        for doc in tqdm(
            nlp.pipe(texts, batch_size=args.batch_size),
            total=len(texts),
            desc=f"corpus {label}",
        ):
            ctx, sents, lemmas, hits = process_document(
                doc, matcher, policy, min_len=min_len, anchor_terms=anchor_terms
            )
            ctx_all.append(ctx)
            sent_all.append(sents)
            lemma_all.append(lemmas)
            anchor_hits.append(hits)

        out = df.copy()
        out["text_pre"] = texts
        out["tokens_ctx"] = ctx_all
        out["sent_idx"] = sent_all
        out["tokens_lemma"] = lemma_all
        out["n_tokens_ctx"] = [len(t) for t in ctx_all]
        out["n_tokens_lemma"] = [len(t) for t in lemma_all]
        out["n_sentences"] = [len(set(s)) for s in sent_all]
        out["n_anchor_hits"] = anchor_hits
        # text_clean kept for continuity with the Phase 1 column name.
        out["text_clean"] = [" ".join(t) for t in lemma_all]

        destination = analysis_dir / f"corpus_{label}.parquet"
        save_table(out, destination, parquet=True, csv=False)

        # CSV twin without the bulky aligned columns.
        slim = out.drop(columns=["tokens_ctx", "sent_idx", "text_pre"])
        save_table(slim, analysis_dir / f"corpus_{label}", parquet=False, csv=True)

        empty_lemma = int((out["n_tokens_lemma"] == 0).sum())
        stats = {
            "n_posts": int(len(out)),
            "subreddits": out["subreddit"].value_counts().to_dict(),
            "total_tokens_lemma": int(out["n_tokens_lemma"].sum()),
            "total_tokens_ctx": int(out["n_tokens_ctx"].sum()),
            "mean_tokens_lemma": round(float(out["n_tokens_lemma"].mean()), 2),
            "mean_sentences": round(float(out["n_sentences"].mean()), 2),
            "total_anchor_hits": int(out["n_anchor_hits"].sum()),
            "posts_with_anchor": int((out["n_anchor_hits"] > 0).sum()),
            "empty_lemma_rate": round(empty_lemma / max(len(out), 1), 4),
            "path": str(destination),
        }
        summary["corpora"][label] = stats

        item(f"tokens_lemma  {stats['total_tokens_lemma']:>7} (mean {stats['mean_tokens_lemma']}/post)")
        item(f"tokens_ctx    {stats['total_tokens_ctx']:>7}")
        item(f"sentences     mean {stats['mean_sentences']}/post")
        item(f"anchor hits   {stats['total_anchor_hits']} across {stats['posts_with_anchor']} posts")
        item(f"empty lemma   {stats['empty_lemma_rate']:.2%}")
        item(f"written       {destination.relative_to(ROOT)}")

    save_summary(summary, analysis_dir / "preprocess_summary.json")
    banner("Done")
    item(f"summary -> {(analysis_dir / 'preprocess_summary.json').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
