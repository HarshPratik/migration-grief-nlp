"""Unit tests for Phase 1 quota and flair-first sampling helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


def _load():
    path = Path(__file__).resolve().parents[1] / "scripts" / "03_filter_and_normalize.py"
    spec = importlib.util.spec_from_file_location("filter_normalize", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_proportional_quotas_sum_and_caps():
    mod = _load()
    q = mod.proportional_quotas({"expats": 12696, "immigration": 9020, "StudyAbroad": 10186}, 10000)
    assert sum(q.values()) == 10000
    assert q["expats"] <= 12696
    assert q["immigration"] <= 9020
    assert q["StudyAbroad"] <= 10186
    assert q["expats"] > q["StudyAbroad"] > q["immigration"]


def test_scarce_first_takes_tiny_subreddit_entirely():
    mod = _load()
    q = mod.scarce_first_quotas({"GriefSupport": 54618, "bereavement": 248}, 10000)
    assert q["bereavement"] == 248
    assert q["GriefSupport"] == 9752
    assert sum(q.values()) == 10000


def test_sample_flair_first_prefers_flair_then_fills():
    mod = _load()
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "post_id": [f"p{i}" for i in range(20)],
            "flair_preferred": [True] * 5 + [False] * 15,
        }
    )
    taken = mod.sample_flair_first(df, 8, rng)
    assert len(taken) == 8
    assert int(taken["flair_preferred"].sum()) == 5
