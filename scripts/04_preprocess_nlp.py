"""
Phase 1 — Clean text, remove stop-words, lemmatize → processed corpora.

  python scripts/04_preprocess_nlp.py
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MD_RE = re.compile(r"[*_~`#>\[\]\(\)]+")
USER_RE = re.compile(r"/?u/[\w-]+", re.I)
SUB_RE = re.compile(r"/?r/[\w-]+", re.I)
HTML_RE = re.compile(r"<[^>]+>")
HTML_ENTITY_RE = re.compile(r"&#\d+;|&[a-z]+;", re.I)
# Reddit RSS/scraper footer noise often appended to body
SUBMITTED_RE = re.compile(
    r"(?:&#32;\s*)*submitted by.*?(?:\[link\]|\[comments\]).*$",
    re.I | re.S,
)
NON_ALPHA_RE = re.compile(r"[^a-zA-Z'\s]+")
WS_RE = re.compile(r"\s+")


def basic_clean(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = SUBMITTED_RE.sub(" ", text)
    t = t.lower()
    t = HTML_RE.sub(" ", t)
    t = HTML_ENTITY_RE.sub(" ", t)
    t = URL_RE.sub(" ", t)
    t = USER_RE.sub(" ", t)
    t = SUB_RE.sub(" ", t)
    t = MD_RE.sub(" ", t)
    t = NON_ALPHA_RE.sub(" ", t)
    t = WS_RE.sub(" ", t).strip()
    return t


def load_spacy():
    import spacy

    try:
        nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
    except OSError:
        from spacy.cli import download

        download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
    return nlp


def lemmatize_batch(texts: list[str], nlp, batch_size: int = 64) -> tuple[list[str], list[list[str]]]:
    clean_strings: list[str] = []
    token_lists: list[list[str]] = []
    for doc in tqdm(
        nlp.pipe((basic_clean(t) for t in texts), batch_size=batch_size),
        total=len(texts),
        desc="lemmatize",
    ):
        lemmas = [
            tok.lemma_.lower()
            for tok in doc
            if not tok.is_stop
            and not tok.is_punct
            and not tok.is_space
            and len(tok.lemma_) >= 2
            and tok.is_alpha
        ]
        token_lists.append(lemmas)
        clean_strings.append(" ".join(lemmas))
    return clean_strings, token_lists


def process_corpus(label: str, nlp) -> pd.DataFrame | None:
    path = INTERIM / f"corpus_{label}.parquet"
    if not path.exists():
        # fallback CSV
        path = INTERIM / f"corpus_{label}.csv"
        if not path.exists():
            print(f"Missing interim corpus {label}: {path}")
            return None
        df = pd.read_csv(path)
    else:
        df = pd.read_parquet(path)

    if df.empty:
        print(f"Corpus {label} is empty — skip")
        return df

    texts = df["text"].fillna("").astype(str).tolist()
    text_clean, tokens = lemmatize_batch(texts, nlp)
    df = df.copy()
    df["text_clean"] = text_clean
    df["tokens"] = tokens
    df["word_count_clean"] = df["tokens"].apply(len)
    return df


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", choices=["a", "b", "both"], default="both")
    args = parser.parse_args()

    PROCESSED.mkdir(parents=True, exist_ok=True)
    nlp = load_spacy()

    labels = ["a", "b"] if args.corpus == "both" else [args.corpus]
    summary: dict = {}

    for label in labels:
        print(f"\n=== Preprocess corpus {label.upper()} ===")
        df = process_corpus(label, nlp)
        if df is None:
            continue
        out_pq = PROCESSED / f"corpus_{label}.parquet"
        out_csv = PROCESSED / f"corpus_{label}.csv"
        # tokens list is fine in parquet; for CSV join as string
        df.to_parquet(out_pq, index=False)
        csv_df = df.copy()
        csv_df["tokens"] = csv_df["tokens"].apply(lambda xs: " ".join(xs) if isinstance(xs, list) else xs)
        csv_df.to_csv(out_csv, index=False)

        empty_rate = float((df["text_clean"].fillna("") == "").mean()) if len(df) else 1.0
        summary[f"corpus_{label}"] = {
            "n_posts": int(len(df)),
            "subreddits": df["subreddit"].value_counts().to_dict() if len(df) else {},
            "mean_word_count": float(df["word_count"].mean()) if len(df) else 0,
            "mean_word_count_clean": float(df["word_count_clean"].mean()) if len(df) else 0,
            "empty_text_clean_rate": empty_rate,
            "date_min": str(df["created_at"].min()) if len(df) else None,
            "date_max": str(df["created_at"].max()) if len(df) else None,
            "path_parquet": str(out_pq),
            "path_csv": str(out_csv),
        }
        print(f"  {len(df)} posts → {out_pq}")
        print(f"  empty text_clean rate: {empty_rate:.2%}")

    summary_path = PROCESSED / "phase1_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"\nSummary → {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
