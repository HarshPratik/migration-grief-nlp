"""Phases 2-4 - corpus loading, output naming, table writing, provenance.

Shared module for scripts 07-21.

Two rules this module enforces so no downstream script has to remember them:

1. `tokens` comes back from parquet as a numpy ndarray, not a list. The
   `isinstance(xs, list)` idiom used at 04_preprocess_nlp.py:133 silently
   no-ops on a re-read. `validate_corpus_frame` normalises it once.
2. Nothing here ever writes to data/processed/. Phase 1 output is frozen;
   `save_table` refuses paths under it.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analysis_config import ANALYSIS_PATH, ANCHORS_PATH, STOPWORDS_PATH, config_sha256  # noqa: E402

PROCESSED_DIR = ROOT / "data" / "processed"
ANALYSIS_DIR = ROOT / "data" / "analysis"
OUTPUTS_DIR = ROOT / "outputs"

CORPUS_LABELS = ("a", "b")

# The 18 columns Phase 1 guarantees. Checked at load so a schema change is
# caught at the boundary rather than as a KeyError 200 lines later.
REQUIRED_COLUMNS = (
    "post_id", "corpus", "subreddit", "title", "body", "text", "created_at",
    "url", "flair", "score", "word_count", "username", "_source_file",
    "flair_preferred", "flair_missing", "text_clean", "tokens", "word_count_clean",
)

# Columns holding sequences; need coercion on read and joining on CSV write.
TOKEN_COLUMNS = ("tokens", "tokens_lemma", "tokens_ctx", "window_tokens", "sent_idx")


class DataError(Exception):
    """Raised when input data does not match the expected contract."""


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------


def as_token_list(value: Any) -> list[str]:
    """Coerce any parquet/CSV token-cell representation to list[str].

    THE trap this project hits: pyarrow `list<string>` reads back as
    numpy.ndarray under pandas 3.x, so `isinstance(value, list)` is False and
    any guard written that way silently does nothing.

    Handles: list, tuple, ndarray, pandas NA/None, float nan, and the
    space-joined string the CSV twin contains.
    """
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return [str(v) for v in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    if isinstance(value, pd.Series):
        return [str(v) for v in value.tolist()]
    if isinstance(value, str):
        return value.split()
    # Scalar NA (np.nan, pd.NA, pd.NaT) - checked last because pd.isna on an
    # array-like returns an array, which is not a valid truth value.
    try:
        if pd.isna(value):
            return []
    except (TypeError, ValueError):
        pass
    return [str(value)]


def validate_corpus_frame(df: pd.DataFrame, label: str, *, required: Sequence[str] = REQUIRED_COLUMNS) -> pd.DataFrame:
    """Validate at the boundary and return a NEW frame with tokens normalised."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DataError(f"corpus {label}: missing column(s) {missing}")
    if df.empty:
        raise DataError(f"corpus {label}: no rows")

    out = df.copy()

    if out["post_id"].duplicated().any():
        dupes = out.loc[out["post_id"].duplicated(), "post_id"].head(5).tolist()
        raise DataError(f"corpus {label}: duplicate post_id(s), e.g. {dupes}")

    if out["text"].isna().all():
        raise DataError(f"corpus {label}: `text` column is entirely null")

    if "created_at" in out.columns:
        raw_created = out["created_at"]
        present = raw_created[raw_created.notna()]
        for value in present:
            try:
                timestamp = pd.Timestamp(value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise DataError(f"corpus {label}: invalid created_at value {value!r}") from exc
            if timestamp.tzinfo is None:
                raise DataError(f"corpus {label}: created_at is not timezone-aware")
        out["created_at"] = pd.to_datetime(raw_created, errors="coerce", utc=True)

    for column in TOKEN_COLUMNS:
        if column in out.columns:
            out[column] = out[column].map(as_token_list)

    return out


def load_corpus(
    label: str,
    *,
    source_dir: Path | None = None,
    validate: bool = True,
) -> pd.DataFrame:
    """Load one corpus parquet. `source_dir` defaults to Phase 1 output."""
    if label not in CORPUS_LABELS:
        raise DataError(f"unknown corpus label {label!r}; expected one of {CORPUS_LABELS}")
    base = source_dir or PROCESSED_DIR
    path = base / f"corpus_{label}.parquet"
    if not path.exists():
        raise DataError(f"corpus {label}: {path} not found")
    df = pd.read_parquet(path)
    return validate_corpus_frame(df, label) if validate else df


def load_corpora(
    labels: Iterable[str] = CORPUS_LABELS,
    *,
    source_dir: Path | None = None,
) -> dict[str, pd.DataFrame]:
    return {label: load_corpus(label, source_dir=source_dir) for label in labels}


# --------------------------------------------------------------------------
# output naming
# --------------------------------------------------------------------------


def out_path(
    kind: str,
    phase: str,
    step: int | str,
    artifact: str,
    *facets: str,
    ext: str = "csv",
    outputs_dir: Path | None = None,
) -> Path:
    """Build one canonical output path. Never f-string these by hand.

    Naming: {phase}_{step:02d}_{artifact}__{facet}__{facet}.{ext}

        out_path("tables", "p2", 10, "neighbors_top15", "a", "ppmi")
        -> outputs/tables/p2_10_neighbors_top15__a__ppmi.csv
    """
    valid = ("tables", "figures", "networks", "summaries", "logs", "bundle")
    if kind not in valid:
        raise ValueError(f"unknown output kind {kind!r}; expected one of {valid}")
    step_txt = f"{step:02d}" if isinstance(step, int) else str(step)
    stem = f"{phase}_{step_txt}_{artifact}"
    for facet in facets:
        if facet:
            stem += f"__{facet}"
    directory = (outputs_dir or OUTPUTS_DIR) / kind
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{stem}.{ext.lstrip('.')}"


# --------------------------------------------------------------------------
# writing
# --------------------------------------------------------------------------


def _csv_safe(df: pd.DataFrame, float_decimals: int | None) -> pd.DataFrame:
    """New frame with sequence columns joined so CSV stays readable."""
    out = df.copy()
    for column in out.columns:
        if column in TOKEN_COLUMNS or out[column].map(lambda v: isinstance(v, (list, np.ndarray))).any():
            out[column] = out[column].map(lambda v: " ".join(str(x) for x in as_token_list(v)))
    if float_decimals is not None:
        for column in out.select_dtypes(include=["float", "float64", "float32"]).columns:
            out[column] = out[column].round(float_decimals)
    return out


def save_table(
    df: pd.DataFrame,
    path: Path,
    *,
    parquet: bool = True,
    csv: bool = True,
    float_decimals: int | None = 4,
    index: bool = False,
) -> list[Path]:
    """Write a table as CSV and/or parquet twins. Does NOT mutate `df`.

    Refuses to write under data/processed/ - Phase 1 output is frozen.
    """
    resolved = path.resolve()
    if PROCESSED_DIR.resolve() in resolved.parents:
        raise DataError(
            f"refusing to write {resolved}: data/processed/ is frozen Phase 1 output. "
            "Phase 2-4 intermediates belong in data/analysis/."
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    if csv:
        csv_path = path.with_suffix(".csv")
        _csv_safe(df, float_decimals).to_csv(csv_path, index=index, encoding="utf-8")
        written.append(csv_path)
    if parquet:
        pq_path = path.with_suffix(".parquet")
        df.to_parquet(pq_path, index=index)
        written.append(pq_path)

    return written


def save_summary(payload: Mapping[str, Any], path: Path, *, add_provenance: bool = True) -> Path:
    """Write a summary JSON with a provenance block attached."""
    resolved = path.resolve()
    if PROCESSED_DIR.resolve() in resolved.parents:
        raise DataError(
            f"refusing to write {resolved}: data/processed/ is frozen Phase 1 output. "
            "Phase 2-4 summaries belong under outputs/."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    body: dict[str, Any] = dict(payload)
    if add_provenance and "provenance" not in body:
        body["provenance"] = provenance()
    path.write_text(json.dumps(body, indent=2, default=str), encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# reproducibility
# --------------------------------------------------------------------------


def make_rng(seed: int) -> np.random.Generator:
    """Explicit Generator, passed around rather than seeding global state."""
    return np.random.default_rng(seed)


def _git_revision() -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        return result.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _package_versions() -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    names = {
        "numpy": "numpy", "pandas": "pandas", "pyarrow": "pyarrow",
        "scipy": "scipy", "scikit-learn": "scikit-learn", "matplotlib": "matplotlib",
        "networkx": "networkx", "spacy": "spacy", "torch": "torch",
        "transformers": "transformers", "vaderSentiment": "vaderSentiment",
    }
    out: dict[str, str] = {}
    for label, dist in names.items():
        try:
            out[label] = version(dist)
        except PackageNotFoundError:
            continue
    return out


def provenance(**extra: Any) -> dict[str, Any]:
    """Everything needed to reproduce or audit an artifact."""
    record: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "script": Path(sys.argv[0]).name if sys.argv and sys.argv[0] else "<interactive>",
        "argv": list(sys.argv),
        "python": sys.version.split()[0],
        "interpreter": sys.executable,
        "config_sha256": config_sha256(ANALYSIS_PATH, ANCHORS_PATH, STOPWORDS_PATH),
        "packages": _package_versions(),
    }
    revision = _git_revision()
    if revision:
        record["git_revision"] = revision
    record.update(extra)
    return record


# --------------------------------------------------------------------------
# console output - matches the Phase 1 print convention, no logging module
# --------------------------------------------------------------------------


def banner(text: str) -> None:
    print(f"\n=== {text} ===")


def item(text: str, indent: int = 1) -> None:
    print(f"{'  ' * indent}{text}")


@dataclass(frozen=True, slots=True)
class Paths:
    """Resolved output roots for one run."""

    analysis: Path
    outputs: Path

    def ensure(self) -> "Paths":
        for directory in (self.analysis, self.outputs):
            directory.mkdir(parents=True, exist_ok=True)
        return self


def resolve_paths(analysis_dir: Path | None = None, outputs_dir: Path | None = None) -> Paths:
    return Paths(
        analysis=analysis_dir or ANALYSIS_DIR,
        outputs=outputs_dir or OUTPUTS_DIR,
    ).ensure()
