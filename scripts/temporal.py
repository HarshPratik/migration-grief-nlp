"""Phase 5 - user-anchored timelines and temporal statistics.

T0 is the author's first recorded post in this sample. That is a proxy for
the origin event in phases/New.docx, not the true date of migration or loss.
Almost all authors appear once, so author-relative bins collapse toward month
0; calendar time is the dual clock that still has mass in later windows.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy import stats

DAYS_PER_MONTH = 365.25 / 12.0
CALENDAR_ORIGIN_DEFAULT = "2023-07-01"
YEAR1_MAX_MONTHS = 12.0
YEAR3_MIN_MONTHS = 24.0


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """BH step-up FDR. NaNs stay NaN; q-values aligned with input order."""
    p = np.asarray(p_values, dtype=float)
    valid = ~np.isnan(p)
    out = np.full(len(p), np.nan)
    if not valid.any():
        return out
    sub = p[valid]
    order = np.argsort(sub)
    ranked = sub[order]
    q = ranked * len(sub) / (np.arange(len(sub)) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    restored = np.empty_like(q)
    restored[order] = np.clip(q, 0, 1)
    out[valid] = restored
    return out


def tag_timelines(
    df: pd.DataFrame,
    *,
    calendar_origin: str = CALENDAR_ORIGIN_DEFAULT,
    year1_max_months: float = YEAR1_MAX_MONTHS,
    year3_min_months: float = YEAR3_MIN_MONTHS,
) -> pd.DataFrame:
    """Tag every document with author-relative and calendar clocks.

    Adds:
      t0_at, days_since_t0, months_since_t0, time_interval (quarter),
      t0_quarter, t0_halfyear, t0_year, n_posts_author, is_singleton,
      is_repeat_author, slice_year1, slice_year3plus,
      cal_days, cal_months, cal_quarter, cal_halfyear, cal_year,
      cal_year1, cal_year3plus
    """
    if "username" not in df.columns or "created_at" not in df.columns:
        raise ValueError("tag_timelines requires username and created_at")
    out = df.copy()
    out["created_at"] = pd.to_datetime(out["created_at"], utc=True, errors="coerce")
    if out["created_at"].isna().any():
        raise ValueError("created_at has unparsable values; refuse to tag")

    t0 = out.groupby("username", sort=False)["created_at"].transform("min")
    out["t0_at"] = t0
    days = (out["created_at"] - t0).dt.total_seconds() / 86400.0
    out["days_since_t0"] = days
    months = days / DAYS_PER_MONTH
    out["months_since_t0"] = months
    out["t0_quarter"] = np.floor(months / 3.0).astype(int)
    out["t0_halfyear"] = np.floor(months / 6.0).astype(int)
    out["t0_year"] = np.floor(months / 12.0).astype(int)
    out["time_interval"] = out["t0_quarter"]

    counts = out.groupby("username", sort=False)["post_id"].transform("size")
    out["n_posts_author"] = counts.astype(int)
    out["is_singleton"] = out["n_posts_author"].eq(1)
    out["is_repeat_author"] = out["n_posts_author"].ge(2)
    out["slice_year1"] = months < year1_max_months
    out["slice_year3plus"] = months >= year3_min_months

    origin = pd.Timestamp(calendar_origin, tz="UTC")
    cal_days = (out["created_at"] - origin).dt.total_seconds() / 86400.0
    out["cal_days"] = cal_days
    out["cal_months"] = cal_days / DAYS_PER_MONTH
    out["cal_quarter"] = np.floor(out["cal_months"] / 3.0).astype(int)
    out["cal_halfyear"] = np.floor(out["cal_months"] / 6.0).astype(int)
    out["cal_year"] = np.floor(out["cal_months"] / 12.0).astype(int)
    out["cal_year1"] = out["cal_months"] < year1_max_months
    out["cal_year3plus"] = out["cal_months"] >= year3_min_months
    return out


def cap_one_per_author(df: pd.DataFrame) -> pd.DataFrame:
    """Deterministic lowest post_id per (corpus, username), matching Phase 3."""
    keys = [c for c in ("corpus", "username") if c in df.columns]
    if "username" not in keys:
        raise ValueError("cap_one_per_author requires username")
    return df.sort_values("post_id").drop_duplicates(keys)


def linregress_row(x: np.ndarray, y: np.ndarray, *, min_n: int = 3) -> dict[str, float]:
    """OLS slope via scipy.stats.linregress. NaN if too few finite pairs."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < min_n or np.unique(x).size < 2:
        return {
            "n": float(len(x)), "slope": np.nan, "intercept": np.nan,
            "rvalue": np.nan, "pvalue": np.nan, "stderr": np.nan,
        }
    result = stats.linregress(x, y)
    return {
        "n": float(len(x)),
        "slope": float(result.slope),
        "intercept": float(result.intercept),
        "rvalue": float(result.rvalue),
        "pvalue": float(result.pvalue),
        "stderr": float(result.stderr),
    }


def theil_sen_row(x: np.ndarray, y: np.ndarray, *, min_n: int = 3) -> dict[str, float]:
    """Robust Theil-Sen slope. NaN if too few finite pairs."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < min_n or np.unique(x).size < 2:
        return {
            "n": float(len(x)), "slope": np.nan, "intercept": np.nan,
            "low": np.nan, "high": np.nan,
        }
    slope, intercept, low, high = stats.theilslopes(y, x)
    return {
        "n": float(len(x)),
        "slope": float(slope),
        "intercept": float(intercept),
        "low": float(low),
        "high": float(high),
    }


def permute_slope_pvalue(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_perm: int,
    rng: np.random.Generator,
    min_n: int = 3,
) -> float:
    """Two-sided permutation p-value for the OLS slope (shuffle y)."""
    observed = linregress_row(x, y, min_n=min_n)
    slope = observed["slope"]
    if not np.isfinite(slope):
        return float("nan")
    y = np.asarray(y, dtype=float)
    count = 0
    for _ in range(int(n_perm)):
        shuffled = rng.permutation(y)
        perm = linregress_row(x, shuffled, min_n=min_n)["slope"]
        if not np.isfinite(perm):
            continue
        if abs(perm) >= abs(slope):
            count += 1
    return (count + 1) / (int(n_perm) + 1)


def cluster_bootstrap_slope(
    frame: pd.DataFrame,
    x_col: str,
    y_col: str,
    cluster_col: str,
    *,
    n_boot: int,
    rng: np.random.Generator,
    min_n: int = 3,
) -> dict[str, float]:
    """Author-clustered bootstrap CI for the OLS slope of y ~ x."""
    x = frame[x_col].to_numpy(dtype=float)
    y = frame[y_col].to_numpy(dtype=float)
    codes, uniques = pd.factorize(frame[cluster_col].astype(str), sort=False)
    observed = linregress_row(x, y, min_n=min_n)
    if uniques.size == 0:
        return {"slope": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n_boot": 0.0}
    grouped_x = [x[codes == i] for i in range(uniques.size)]
    grouped_y = [y[codes == i] for i in range(uniques.size)]
    n_clusters = uniques.size
    slopes: list[float] = []
    for _ in range(int(n_boot)):
        draw = rng.integers(0, n_clusters, size=n_clusters)
        xb = np.concatenate([grouped_x[i] for i in draw])
        yb = np.concatenate([grouped_y[i] for i in draw])
        slope = linregress_row(xb, yb, min_n=min_n)["slope"]
        if np.isfinite(slope):
            slopes.append(float(slope))
    if not slopes:
        return {
            "slope": observed["slope"], "ci_low": np.nan, "ci_high": np.nan,
            "n_boot": 0.0, "pvalue": observed["pvalue"],
        }
    arr = np.asarray(slopes, dtype=float)
    return {
        "slope": float(observed["slope"]),
        "ci_low": float(np.quantile(arr, 0.025)),
        "ci_high": float(np.quantile(arr, 0.975)),
        "n_boot": float(len(arr)),
        "pvalue": float(observed["pvalue"]),
    }


def bin_means(
    frame: pd.DataFrame,
    *,
    bin_col: str,
    value_col: str,
    min_bin_posts: int,
) -> pd.DataFrame:
    """Mean of value_col per integer bin, dropping thin bins."""
    grouped = (
        frame.groupby(bin_col, dropna=True)[value_col]
        .agg(mean="mean", std="std", n="size")
        .reset_index()
        .rename(columns={bin_col: "bin"})
    )
    grouped = grouped[grouped["n"] >= int(min_bin_posts)].copy()
    grouped["bin"] = grouped["bin"].astype(int)
    return grouped.sort_values("bin")


def interpret_sadness_slope(slope: float, pvalue: float, q: float, alpha: float = 0.05) -> str:
    """Map New.docx's decay vs stationarity rule onto a slope test."""
    if not np.isfinite(slope) or not np.isfinite(pvalue):
        return "underpowered"
    if q <= alpha and slope < 0:
        return "decay"
    if q <= alpha and slope > 0:
        return "increase"
    return "stationary_or_null"


def temporal_power_audit(df: pd.DataFrame, *, corpus: str) -> dict[str, Any]:
    """Honest sample-size numbers for the paper methods section."""
    n_posts = int(len(df))
    n_authors = int(df["username"].nunique())
    n_repeat_authors = int(df.loc[df["is_repeat_author"], "username"].nunique())
    n_repeat_posts = int(df["is_repeat_author"].sum())
    return {
        "corpus": corpus,
        "n_posts": n_posts,
        "n_authors": n_authors,
        "n_repeat_authors": n_repeat_authors,
        "n_repeat_posts": n_repeat_posts,
        "singleton_author_frac": float((df.groupby("username").size() == 1).mean()),
        "repeat_post_frac": float(n_repeat_posts / n_posts) if n_posts else 0.0,
        "t0_quarter_counts": df["t0_quarter"].value_counts().sort_index().to_dict(),
        "cal_quarter_counts": df["cal_quarter"].value_counts().sort_index().to_dict(),
        "year1_posts": int(df["slice_year1"].sum()),
        "year3plus_posts": int(df["slice_year3plus"].sum()),
        "cal_year1_posts": int(df["cal_year1"].sum()),
        "cal_year3plus_posts": int(df["cal_year3plus"].sum()),
        "t0_note": (
            "T0 is first post in this sample, not the origin event. "
            "Singletons all sit at time_interval=0."
        ),
    }


def cfg_temporal(cfg: Mapping[str, Any] | Any) -> dict[str, Any]:
    """Read temporal knobs with defaults so tests need not ship the full yaml."""
    get = cfg.get if hasattr(cfg, "get") else lambda key, default=None: default
    return {
        "calendar_origin": str(get("temporal.calendar_origin", CALENDAR_ORIGIN_DEFAULT)),
        "year1_max_months": float(get("temporal.year1_max_months", YEAR1_MAX_MONTHS)),
        "year3_min_months": float(get("temporal.year3_min_months", YEAR3_MIN_MONTHS)),
        "min_bin_posts": int(get("temporal.min_bin_posts", 30)),
        "n_bootstrap": int(get("temporal.n_bootstrap", 2000)),
        "n_permutations": int(get("temporal.n_permutations", 2000)),
        "betweenness_spike_ratio": float(get("temporal.betweenness_spike_ratio", 2.0)),
        "topic_k": int(get("temporal.topic_k", 12)),
        "minilm_model": str(get("temporal.minilm_model", "all-MiniLM-L6-v2")),
        "fullpost_max_length": int(get("temporal.fullpost_max_length", 256)),
    }
