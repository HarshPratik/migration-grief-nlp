"""Phases 2-4 - verify the .venv-analysis environment is intact.

This is a permanent gate, not a throwaway diagnostic. Run it after every
install into .venv-analysis. It exits non-zero on the first hard failure.

The check that matters most is the numpy ABI canary (check 3): spacy, thinc and
blis are compiled against numpy 2.x. If some package quietly downgraded numpy to
1.x, `import spacy` still succeeds but `spacy.load()` raises a cryptic
"_ARRAY_API not found". Catching that here turns a confusing three-scripts-later
failure into an obvious one.

Example:
    D:\\research_paper\\.venv-analysis\\Scripts\\python.exe scripts/check_env.py
    D:\\research_paper\\.venv-analysis\\Scripts\\python.exe scripts/check_env.py --skip-corpus
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Versions the Phase 1 binary stack is built against. A mismatch here is the
# downgrade this whole constraints-file strategy exists to prevent.
PINNED = {
    "numpy": "2.5.1",
    "pandas": "3.0.3",
    "pyarrow": "25.0.0",
    "spacy": "3.8.14",
}

# Packages deliberately excluded - see requirements-analysis.txt for reasons.
# Their presence means someone installed outside the constraints file.
BANNED = ("gensim", "umap", "hdbscan", "bertopic", "numba")

REPORT_PACKAGES = (
    "numpy", "pandas", "pyarrow", "scipy", "sklearn", "matplotlib",
    "networkx", "spacy", "vaderSentiment", "torch", "transformers",
    "sentence_transformers", "openpyxl", "pytest",
)

failures: list[str] = []
warnings: list[str] = []


def check(label: str, fn) -> None:
    """Run one check, recording rather than raising so all failures surface."""
    try:
        detail = fn()
    except Exception as exc:  # noqa: BLE001 - a failed check must not abort the run
        failures.append(f"{label}: {type(exc).__name__}: {exc}")
        print(f"  FAIL  {label}")
        print(f"        {type(exc).__name__}: {exc}")
        return
    print(f"  ok    {label}" + (f" - {detail}" if detail else ""))


# Import name -> distribution name, where they differ.
DIST_NAMES = {
    "sklearn": "scikit-learn",
    "sentence_transformers": "sentence-transformers",
}


def _version(module_name: str) -> str:
    """Prefer installed-distribution metadata; not every package exposes __version__."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(DIST_NAMES.get(module_name, module_name))
    except PackageNotFoundError:
        mod = importlib.import_module(module_name)
        return getattr(mod, "__version__", "unknown")


def check_interpreter() -> str:
    exe = Path(sys.executable).resolve()
    expected = (ROOT / ".venv-analysis").resolve()
    if expected not in exe.parents:
        # A warning, not a failure: the user may deliberately be testing another env.
        warnings.append(
            f"interpreter is {exe}, not under {expected}. "
            "Scripts 07+ expect .venv-analysis; 01-06 use .venv."
        )
    return f"{exe} (Python {sys.version.split()[0]})"


def check_pinned_versions() -> str:
    bad = []
    for name, want in PINNED.items():
        got = _version(name)
        if got != want:
            bad.append(f"{name}={got} (expected {want})")
    if bad:
        raise AssertionError(
            "version drift - something resolved outside constraints-analysis.txt: "
            + ", ".join(bad)
        )
    return ", ".join(f"{k}=={v}" for k, v in PINNED.items())


def check_banned_absent() -> str:
    import importlib.util

    found = [n for n in BANNED if importlib.util.find_spec(n) is not None]
    if found:
        raise AssertionError(
            f"deliberately-excluded package(s) installed: {found}. "
            "These pin numpy<2 and will break the spacy binary stack."
        )
    return "none of " + "/".join(BANNED)


def check_spacy_sm() -> str:
    """The numpy ABI canary. A downgraded numpy fails here, loudly."""
    import spacy

    nlp = spacy.load("en_core_web_sm")
    doc = nlp("The children were grieving quietly at home.")
    assert len(doc) >= 5, f"expected 5+ tokens, got {len(doc)}"
    assert doc[1].lemma_ == "child", f"lemmatizer wrong: {doc[1].lemma_!r} != 'child'"
    return f"{len(doc)} tokens, lemma ok"


def check_spacy_md_vectors() -> str:
    import spacy

    nlp = spacy.load("en_core_web_md")
    assert nlp.vocab["home"].has_vector, "en_core_web_md has no vector for 'home'"
    rows, dims = nlp.vocab.vectors.shape
    assert dims == 300, f"expected 300-d vectors, got {dims}"
    return f"{rows} keys x {dims}d"


def check_senter() -> str:
    """Stage 2 needs sentence boundaries; Phase 1 disabled parser and senter."""
    import spacy

    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
    # en_core_web_sm ships `senter` disabled (the parser normally sets sentence
    # boundaries). `pipe_names` lists only ENABLED components, so check
    # `component_names`, which includes disabled ones, before adding.
    if "senter" in nlp.component_names:
        nlp.enable_pipe("senter")
    else:
        nlp.add_pipe("senter")
    sents = list(nlp("I moved away. I miss home. It still hurts.").sents)
    assert len(sents) == 3, f"expected 3 sentences, got {len(sents)}"
    return "3/3 sentences"


def check_corpus_roundtrip() -> str:
    """Pin the known parquet trap: `tokens` reads back as ndarray, not list."""
    import pandas as pd

    path = ROOT / "data" / "processed" / "corpus_a.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run Phase 1 first, or pass --skip-corpus")
    df = pd.read_parquet(path)
    assert len(df.columns) == 18, f"expected 18 columns, got {len(df.columns)}"
    kind = type(df["tokens"].iloc[0]).__name__
    assert kind == "ndarray", (
        f"`tokens` came back as {kind}, not ndarray. The as_token_list() helper in "
        "scripts/text_windows.py assumes ndarray; re-check it before trusting results."
    )
    assert df["created_at"].dt.tz is not None, "created_at lost its timezone"
    return f"{len(df)} rows x {len(df.columns)} cols, tokens={kind}"


def check_sklearn_scipy() -> str:
    import numpy as np
    import scipy.sparse as sp
    from sklearn.decomposition import PCA, TruncatedSVD

    rng = np.random.default_rng(0)
    PCA(n_components=3).fit(rng.random((50, 10)))
    mat = sp.csr_matrix(rng.random((40, 20)))
    TruncatedSVD(n_components=5, random_state=0).fit(mat)
    return "PCA + sparse TruncatedSVD"


def check_matplotlib() -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(2, 2))
    ax.plot([0, 1], [0, 1])
    tmp = Path(tempfile.gettempdir()) / "_check_env_probe.png"
    fig.savefig(tmp, dpi=72)
    plt.close(fig)
    try:
        size = tmp.stat().st_size
        assert size > 0, "rendered PNG is empty"
    finally:
        tmp.unlink(missing_ok=True)
    return f"Agg backend, {size} byte PNG"


def check_vader() -> str:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    an = SentimentIntensityAnalyzer()
    pos = an.polarity_scores("I love this")["compound"]
    neg = an.polarity_scores("I do not love this")["compound"]
    assert pos > neg, f"negation not handled: {pos} !> {neg}"
    return f"negation ok ({pos:+.3f} vs {neg:+.3f})"


def check_torch_optional() -> str:
    """Optional until the transformer layer is installed - but if present,
    it must be the CPU build. A CUDA wheel is ~2.5 GB of wasted disk."""
    import importlib.util

    if importlib.util.find_spec("torch") is None:
        warnings.append(
            "torch not installed yet - Stage 6 (emotion classification) needs it. "
            "Install: pip install --index-url https://download.pytorch.org/whl/cpu torch"
        )
        return "not installed (optional until Stage 6)"
    import torch

    if torch.cuda.is_available():
        raise AssertionError(
            "CUDA torch detected. Expected the CPU-only build - reinstall with "
            "--index-url https://download.pytorch.org/whl/cpu"
        )
    return f"{torch.__version__} (CPU)"


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the .venv-analysis environment")
    parser.add_argument(
        "--skip-corpus",
        action="store_true",
        help="skip the data/processed round-trip check (useful before Phase 1 has run)",
    )
    args = parser.parse_args()

    print("=== Environment check ===")
    check("interpreter", check_interpreter)
    check("pinned versions", check_pinned_versions)
    check("excluded packages absent", check_banned_absent)
    check("spacy en_core_web_sm (numpy ABI canary)", check_spacy_sm)
    check("spacy en_core_web_md vectors", check_spacy_md_vectors)
    check("spacy sentence segmentation", check_senter)
    check("scipy + sklearn", check_sklearn_scipy)
    check("matplotlib Agg render", check_matplotlib)
    check("vaderSentiment", check_vader)
    check("torch (optional)", check_torch_optional)
    if not args.skip_corpus:
        check("corpus_a.parquet round-trip", check_corpus_roundtrip)

    print("\n=== Versions ===")
    import importlib.util

    for name in REPORT_PACKAGES:
        if importlib.util.find_spec(name) is None:
            print(f"  {name:24} -")
            continue
        try:
            print(f"  {name:24} {_version(name)}")
        except Exception as exc:  # noqa: BLE001
            print(f"  {name:24} ? ({type(exc).__name__})")

    if warnings:
        print("\n=== Warnings ===")
        for w in warnings:
            print(f"  ! {w}")

    print()
    if failures:
        print(f"=== FAILED ({len(failures)}) ===")
        for f in failures:
            print(f"  x {f}")
        return 1
    print("=== All checks passed ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
