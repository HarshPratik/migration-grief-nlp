"""Import every numbered pipeline script and check it exposes `main`.

The numbered scripts cannot be imported by name - `01_scrape_apify` is not a
valid identifier - so they are loaded from their file path. Cheap as this is, it
is the only automated check that catches a syntax error or a missing dependency
in scripts 01-21 before someone runs a two-hour job.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"


def _pipeline_scripts() -> list[Path]:
    """Numbered pipeline scripts plus check_env; `_*.py` throwaways excluded."""
    numbered = [p for p in SCRIPTS_DIR.glob("*.py") if p.name[0].isdigit()]
    env = SCRIPTS_DIR / "check_env.py"
    if env.exists():
        numbered.append(env)
    return sorted(numbered, key=lambda p: p.name)


SCRIPTS = _pipeline_scripts()


def _load(path: Path):
    name = "pipeline_" + path.stem.replace(".", "_")
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pipeline_scripts_were_discovered():
    names = {p.name for p in SCRIPTS}
    assert "check_env.py" in names
    assert len([n for n in names if n[0].isdigit()]) >= 9
    assert not any(n.startswith("_") for n in names)


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: p.name)
def test_script_imports_and_exposes_main(path: Path):
    module = _load(path)
    assert callable(getattr(module, "main", None)), f"{path.name} has no callable main()"
