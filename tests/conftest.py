"""Shared fixtures for the Phase 2-4 analysis-module test suite.

The five modules under test live in `scripts/`, which is not a package: there is
no `__init__.py` and several sibling filenames start with a digit
(`09_anchor_coverage.py`), so `import scripts.vectors` can never work. The
production scripts themselves do `sys.path.insert(0, ROOT / "scripts")`; the
tests do exactly the same thing so they exercise the real import path.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # no window, no display, safe in CI


# --------------------------------------------------------------------------
# corpus frame
# --------------------------------------------------------------------------

# The 18 columns Phase 1 guarantees, in the order analysis_io declares them.
CORPUS_COLUMNS = (
    "post_id", "corpus", "subreddit", "title", "body", "text", "created_at",
    "url", "flair", "score", "word_count", "username", "_source_file",
    "flair_preferred", "flair_missing", "text_clean", "tokens", "word_count_clean",
)

_SAMPLE_TOKENS = (
    ("timezone", "weekend", "miss", "home"),
    ("video_call", "camera", "screen", "blurry"),
    ("gone", "empty", "alone", "still"),
    ("name", "identity", "passport", "papers"),
)


def make_corpus_frame(n_rows: int = 20, label: str = "a") -> pd.DataFrame:
    """A synthetic Phase 1 corpus frame whose `tokens` cells are numpy arrays.

    Parquet hands `tokens` back as `numpy.ndarray`, never `list`. Building the
    fixture the same way is the whole point: a test built from plain lists would
    pass against code that only handles lists.
    """
    tokens = pd.Series(
        [np.array(_SAMPLE_TOKENS[i % len(_SAMPLE_TOKENS)], dtype=object) for i in range(n_rows)],
        dtype=object,
    )
    texts = [" ".join(_SAMPLE_TOKENS[i % len(_SAMPLE_TOKENS)]) for i in range(n_rows)]
    return pd.DataFrame(
        {
            "post_id": [f"{label}{i:03d}" for i in range(n_rows)],
            "corpus": [label] * n_rows,
            "subreddit": ["expats" if label == "a" else "GriefSupport"] * n_rows,
            "title": [f"title {i}" for i in range(n_rows)],
            "body": [f"body text number {i}" for i in range(n_rows)],
            "text": texts,
            "created_at": pd.date_range("2024-01-01", periods=n_rows, freq="D", tz="UTC"),
            "url": [f"https://reddit.com/r/x/comments/{i}" for i in range(n_rows)],
            "flair": ["Question" if i % 3 else None for i in range(n_rows)],
            "score": np.arange(n_rows, dtype="int64"),
            "word_count": [len(t.split()) for t in texts],
            "username": [f"user_{i % 7}" for i in range(n_rows)],
            "_source_file": ["synthetic.jsonl"] * n_rows,
            "flair_preferred": [bool(i % 3) for i in range(n_rows)],
            "flair_missing": [not bool(i % 3) for i in range(n_rows)],
            "text_clean": texts,
            "tokens": tokens,
            "word_count_clean": [len(t.split()) for t in texts],
        }
    )


@pytest.fixture
def corpus_df() -> pd.DataFrame:
    """20-row synthetic corpus with all 18 required columns."""
    return make_corpus_frame(20, "a")


# --------------------------------------------------------------------------
# YAML factories - always under tmp_path, never the real config/
# --------------------------------------------------------------------------


@pytest.fixture
def write_yaml(tmp_path: Path):
    """Dump a Python object to a YAML file under tmp_path and return the path."""

    def _write(name: str, data: object) -> Path:
        path = tmp_path / name
        path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        return path

    return _write


@pytest.fixture
def write_text_yaml(tmp_path: Path):
    """Write raw YAML source - needed to reproduce parser-level traps verbatim."""

    def _write(name: str, text: str) -> Path:
        path = tmp_path / name
        path.write_text(text, encoding="utf-8")
        return path

    return _write


def base_anchors() -> dict:
    """A minimal but fully valid anchors config, ready to be broken per-test."""
    return {
        "version": 1,
        "defaults": {"match": "surface", "case_sensitive": False, "tier": "rare"},
        "domains": {
            "temporal": {
                "label": "Temporal Disruption",
                "subtitle": "Umwelt",
                "color": "#4C78A8",
                "description": "time and distance",
                "anchors": [
                    {"term": "timezone", "surface_forms": ["timezone", "time zone"]},
                    {"term": "weekend", "surface_forms": ["weekend", "weekends"]},
                ],
            },
            "absence": {
                "label": "Absence",
                "color": "#E45756",
                "description": "loss",
                "anchors": [
                    {"term": "gone", "surface_forms": ["gone"], "tier": "mid"},
                ],
            },
        },
        "comparisons": [
            {
                "name": "temporal_vs_absence",
                "left": "temporal",
                "right": "absence",
                "hypothesis": "closer in B",
            }
        ],
        "neighbour_blocklist": ["reddit", "OP"],
    }


def base_analysis() -> dict:
    """A minimal but fully valid analysis config."""
    return {
        "version": 1,
        "seed": 20260726,
        "paths": {
            "input_dir": "data/processed",
            "analysis_dir": "data/analysis",
            "outputs_dir": "outputs",
        },
        "tiers": {
            "tier1": {"min_freq": 50, "min_docfreq": 25},
            "tier2": {"min_freq": 20, "min_docfreq": 10},
        },
        "windows": {"size": 5, "clip_at_sentence": True, "negation_rescue": 4},
        "figures": {"dpi": 300, "width_in": 8.0, "height_in": 5.0},
    }


def base_stopwords() -> dict:
    """A minimal valid stopword policy that needs no spaCy import."""
    return {
        "version": 1,
        "source": "none",
        "protect": ["not", "no", "never"],
        "add": ["reddit", "tldr"],
        "remove": [],
    }


@pytest.fixture
def anchors_data() -> dict:
    return base_anchors()


@pytest.fixture
def analysis_data() -> dict:
    return base_analysis()


@pytest.fixture
def stopwords_data() -> dict:
    return base_stopwords()


# --------------------------------------------------------------------------
# real config, loaded once - load_stopword_policy imports spaCy (~4s cold)
# --------------------------------------------------------------------------


@pytest.fixture(scope="session")
def real_anchors():
    from analysis_config import load_anchors

    return load_anchors()


@pytest.fixture(scope="session")
def real_stopwords():
    """Load real stopwords.yaml against a representative spaCy base set.

    Importing spaCy solely to obtain its constant STOP_WORDS set costs more than
    five seconds on a cold Windows process. The policy's own contract is set
    arithmetic, so a controlled base containing a true stopword ("the") plus
    words that the real config protects tests that contract without turning a
    fast unit suite into an NLP-model integration test.
    """
    from types import ModuleType
    from unittest.mock import patch

    from analysis_config import load_stopword_policy

    spacy = ModuleType("spacy")
    lang = ModuleType("spacy.lang")
    english = ModuleType("spacy.lang.en")
    stop_words = ModuleType("spacy.lang.en.stop_words")
    stop_words.STOP_WORDS = frozenset({
        "the", "not", "no", "never", "alone", "gone", "miss", "ordinary_stop",
    })
    modules = {
        "spacy": spacy,
        "spacy.lang": lang,
        "spacy.lang.en": english,
        "spacy.lang.en.stop_words": stop_words,
    }
    with patch.dict(sys.modules, modules):
        return load_stopword_policy()


# --------------------------------------------------------------------------
# figure style
# --------------------------------------------------------------------------


@pytest.fixture
def style():
    from plotting import apply_style

    return apply_style()
