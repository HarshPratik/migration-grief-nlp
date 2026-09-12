"""Unit tests for scripts/temporal.py — author-relative T0 and slope helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from temporal import (
    DAYS_PER_MONTH,
    benjamini_hochberg,
    bin_means,
    cap_one_per_author,
    interpret_sadness_slope,
    linregress_row,
    permute_slope_pvalue,
    tag_timelines,
    theil_sen_row,
    temporal_power_audit,
)


def _posts() -> pd.DataFrame:
    """Two authors: one singleton at T0, one with posts at 0 / 6 / 30 months."""
    t0 = pd.Timestamp("2023-07-01", tz="UTC")
    rows = [
        ("p1", "a", "alice", t0),
        ("p2", "a", "bob", t0),
        ("p3", "a", "bob", t0 + pd.DateOffset(months=6)),
        ("p4", "a", "bob", t0 + pd.DateOffset(months=30)),
    ]
    return pd.DataFrame(
        {
            "post_id": [r[0] for r in rows],
            "corpus": [r[1] for r in rows],
            "username": [r[2] for r in rows],
            "created_at": [r[3] for r in rows],
            "text": ["x"] * 4,
            "word_count": [120] * 4,
        }
    )


def test_singleton_sits_at_interval_zero():
    tagged = tag_timelines(_posts())
    alice = tagged[tagged["username"] == "alice"].iloc[0]
    assert alice["time_interval"] == 0
    assert alice["months_since_t0"] == 0
    assert bool(alice["is_singleton"]) is True
    assert bool(alice["slice_year1"]) is True
    assert bool(alice["slice_year3plus"]) is False


def test_repeat_author_bins_and_year_slices():
    tagged = tag_timelines(_posts())
    bob = tagged[tagged["username"] == "bob"].sort_values("created_at")
    months = bob["months_since_t0"].to_numpy()
    assert months[0] == pytest.approx(0.0)
    assert months[1] == pytest.approx(6.0, abs=0.2)
    assert months[2] == pytest.approx(30.0, abs=0.3)
    assert list(bob["t0_quarter"]) == [0, 2, 10]
    assert list(bob["t0_halfyear"]) == [0, 1, 5]
    flags = bob[["slice_year1", "slice_year3plus"]].astype(bool)
    assert list(flags["slice_year1"]) == [True, True, False]
    assert list(flags["slice_year3plus"]) == [False, False, True]
    assert int(bob["n_posts_author"].iloc[0]) == 3


def test_calendar_clock_uses_phase1_origin():
    tagged = tag_timelines(_posts(), calendar_origin="2023-07-01")
    assert tagged["cal_quarter"].min() == 0
    late = tagged.loc[tagged["post_id"] == "p4"].iloc[0]
    assert bool(late["cal_year3plus"]) is True


def test_refuses_unparsable_timestamps():
    df = _posts()
    values = df["created_at"].astype(str).tolist()
    values[0] = "not-a-date"
    df["created_at"] = values
    with pytest.raises(ValueError, match="unparsable"):
        tag_timelines(df)


def test_cap_one_keeps_lowest_post_id():
    df = _posts()
    capped = cap_one_per_author(df)
    assert set(capped["username"]) == {"alice", "bob"}
    assert capped.loc[capped["username"] == "bob", "post_id"].iloc[0] == "p2"


def test_linregress_negative_slope_and_underpowered():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([4.0, 3.0, 2.0, 1.0])
    row = linregress_row(x, y)
    assert row["slope"] == pytest.approx(-1.0)
    assert row["pvalue"] < 0.05
    empty = linregress_row(np.array([1.0, 1.0]), np.array([2.0, 3.0]))
    assert np.isnan(empty["slope"])
    robust = theil_sen_row(x, y)
    assert robust["slope"] == pytest.approx(-1.0)


def test_permutation_pvalue_is_small_for_strong_slope():
    rng = np.random.default_rng(0)
    x = np.arange(12, dtype=float)
    y = -2.0 * x
    p = permute_slope_pvalue(x, y, n_perm=200, rng=rng)
    assert p < 0.05


def test_bin_means_drops_thin_bins():
    frame = pd.DataFrame({"bin": [0, 0, 0, 1, 1, 2], "val": [1.0, 2.0, 3.0, 4.0, 5.0, 9.0]})
    out = bin_means(frame, bin_col="bin", value_col="val", min_bin_posts=3)
    assert list(out["bin"]) == [0]
    assert out["mean"].iloc[0] == pytest.approx(2.0)


def test_interpret_and_fdr_and_audit():
    assert interpret_sadness_slope(-0.1, 0.001, 0.01) == "decay"
    assert interpret_sadness_slope(0.0, 0.8, 0.9) == "stationary_or_null"
    assert interpret_sadness_slope(np.nan, np.nan, np.nan) == "underpowered"
    q = benjamini_hochberg(np.array([0.01, 0.04, 0.5]))
    assert q[0] <= q[1] <= q[2]
    tagged = tag_timelines(_posts())
    audit = temporal_power_audit(tagged, corpus="a")
    assert audit["n_authors"] == 2
    assert audit["n_repeat_authors"] == 1
    assert audit["year3plus_posts"] == 1
    assert DAYS_PER_MONTH == pytest.approx(30.4375)
