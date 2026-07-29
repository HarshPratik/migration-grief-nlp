"""Tests for scripts/analysis_io.py.

`as_token_list` gets the most attention here because it is the single point
where the project's most expensive bug class is contained: pyarrow returns a
`list<string>` column as `numpy.ndarray`, so every `isinstance(x, list)` guard
written against it silently does nothing and the failure shows up hundreds of
lines later as an empty frequency table.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis_io import (
    DataError,
    Paths,
    as_token_list,
    load_corpus,
    make_rng,
    out_path,
    provenance,
    resolve_paths,
    save_summary,
    save_table,
    validate_corpus_frame,
)


# --------------------------------------------------------------------------
# as_token_list - the highest-value function in the project
# --------------------------------------------------------------------------


def test_as_token_list_numpy_array_of_str():
    # The parquet round-trip shape. `isinstance(value, list)` is False here.
    value = np.array(["home", "sick"], dtype=object)
    assert not isinstance(value, list)
    assert as_token_list(value) == ["home", "sick"]


def test_as_token_list_empty_ndarray():
    assert as_token_list(np.array([], dtype=object)) == []


def test_as_token_list_list_passthrough():
    assert as_token_list(["a", "b"]) == ["a", "b"]


def test_as_token_list_tuple():
    assert as_token_list(("a", "b")) == ["a", "b"]


def test_as_token_list_space_joined_string():
    # What the CSV twin contains after _csv_safe joins the sequence.
    assert as_token_list("a b  c") == ["a", "b", "c"]


def test_as_token_list_none():
    assert as_token_list(None) == []


def test_as_token_list_numpy_nan():
    assert as_token_list(np.nan) == []


def test_as_token_list_pandas_na():
    assert as_token_list(pd.NA) == []


def test_as_token_list_empty_list():
    assert as_token_list([]) == []


def test_as_token_list_float_scalar_becomes_one_token():
    # A stray numeric cell must not raise; it degrades to a single token so the
    # caller sees the junk rather than an exception mid-pipeline.
    assert as_token_list(3.5) == ["3.5"]


def test_as_token_list_always_returns_str_elements():
    assert as_token_list(np.array([1, 2])) == ["1", "2"]


def test_as_token_list_does_not_mutate_input():
    source = ["a", "b"]
    result = as_token_list(source)
    result.append("c")
    assert source == ["a", "b"]


# --------------------------------------------------------------------------
# out_path
# --------------------------------------------------------------------------


def test_out_path_naming_shape(tmp_path):
    path = out_path("tables", "p2", 10, "neighbors_top15", "a", "ppmi", outputs_dir=tmp_path)
    assert path.name == "p2_10_neighbors_top15__a__ppmi.csv"
    assert path.parent == tmp_path / "tables"
    assert path.parent.is_dir()


def test_out_path_zero_pads_integer_step(tmp_path):
    path = out_path("figures", "p3", 7, "emotion", ext="png", outputs_dir=tmp_path)
    assert path.name == "p3_07_emotion.png"


def test_out_path_skips_empty_facets(tmp_path):
    path = out_path("tables", "p2", 9, "coverage", "", "b", outputs_dir=tmp_path)
    assert path.name == "p2_09_coverage__b.csv"


def test_out_path_rejects_unknown_kind(tmp_path):
    with pytest.raises(ValueError, match="unknown output kind"):
        out_path("spreadsheets", "p2", 1, "x", outputs_dir=tmp_path)


# --------------------------------------------------------------------------
# save_table
# --------------------------------------------------------------------------


def test_save_table_writes_csv_and_parquet_twins(tmp_path, corpus_df):
    written = save_table(corpus_df, tmp_path / "corpus")
    assert [p.suffix for p in written] == [".csv", ".parquet"]
    assert all(p.exists() and p.stat().st_size > 0 for p in written)


def test_save_table_does_not_mutate_caller_frame(corpus_df, tmp_path):
    # The immutability rule, enforced mechanically: _csv_safe joins token
    # sequences into strings, and must do so on a copy.
    before = corpus_df.at[0, "tokens"]
    assert isinstance(before, np.ndarray)

    save_table(corpus_df, tmp_path / "t")

    after = corpus_df.at[0, "tokens"]
    assert isinstance(after, np.ndarray), "save_table mutated the caller's tokens cell"
    assert list(after) == list(before)


def test_save_table_csv_joins_token_sequences(tmp_path, corpus_df):
    (csv_path,) = save_table(corpus_df, tmp_path / "t", parquet=False)
    text = csv_path.read_text(encoding="utf-8")
    assert "timezone weekend miss home" in text


def test_save_table_rounds_floats(tmp_path):
    df = pd.DataFrame({"x": [1.23456789]})
    (csv_path,) = save_table(df, tmp_path / "t", parquet=False, float_decimals=3)
    assert "1.235" in csv_path.read_text(encoding="utf-8")


def test_save_table_refuses_data_processed(tmp_path):
    from analysis_io import PROCESSED_DIR

    df = pd.DataFrame({"x": [1]})
    with pytest.raises(DataError, match="frozen Phase 1 output"):
        save_table(df, PROCESSED_DIR / "sub" / "sneaky.csv")


def test_save_table_csv_only(tmp_path):
    written = save_table(pd.DataFrame({"x": [1]}), tmp_path / "t", parquet=False)
    assert len(written) == 1 and written[0].suffix == ".csv"


# --------------------------------------------------------------------------
# save_summary / provenance / rng
# --------------------------------------------------------------------------


def test_save_summary_refuses_data_processed():
    from analysis_io import PROCESSED_DIR

    with pytest.raises(DataError, match="frozen Phase 1 output"):
        save_summary({"n": 3}, PROCESSED_DIR / "sneaky.json")


def test_save_summary_attaches_provenance(tmp_path):
    import json

    path = save_summary({"n": 3}, tmp_path / "s.json")
    body = json.loads(path.read_text(encoding="utf-8"))
    assert body["n"] == 3
    assert "provenance" in body


def test_provenance_has_required_keys():
    record = provenance()
    for key in ("timestamp_utc", "argv", "config_sha256", "packages"):
        assert key in record, f"provenance() missing {key!r}"
    assert isinstance(record["argv"], list)
    assert len(record["config_sha256"]) == 64


def test_provenance_accepts_extra_fields():
    assert provenance(corpus="a")["corpus"] == "a"


def test_make_rng_is_deterministic():
    assert make_rng(42).random(5).tolist() == make_rng(42).random(5).tolist()


def test_make_rng_differs_across_seeds():
    assert make_rng(1).random(5).tolist() != make_rng(2).random(5).tolist()


# --------------------------------------------------------------------------
# validate_corpus_frame
# --------------------------------------------------------------------------


def test_validate_corpus_frame_accepts_valid_frame(corpus_df):
    out = validate_corpus_frame(corpus_df, "a")
    assert len(out) == len(corpus_df)


def test_validate_corpus_frame_coerces_token_columns(corpus_df):
    out = validate_corpus_frame(corpus_df, "a")
    assert isinstance(out.at[0, "tokens"], list)
    assert out.at[0, "tokens"] == ["timezone", "weekend", "miss", "home"]


def test_validate_corpus_frame_returns_new_frame(corpus_df):
    out = validate_corpus_frame(corpus_df, "a")
    assert out is not corpus_df
    assert isinstance(corpus_df.at[0, "tokens"], np.ndarray)


def test_validate_corpus_frame_missing_column(corpus_df):
    broken = corpus_df.drop(columns=["word_count_clean"])
    with pytest.raises(DataError, match="missing column"):
        validate_corpus_frame(broken, "a")


def test_validate_corpus_frame_duplicate_post_id(corpus_df):
    broken = corpus_df.copy()
    broken.loc[broken.index[1], "post_id"] = broken.at[0, "post_id"]
    with pytest.raises(DataError, match="duplicate post_id"):
        validate_corpus_frame(broken, "a")


def test_validate_corpus_frame_rejects_empty(corpus_df):
    with pytest.raises(DataError, match="no rows"):
        validate_corpus_frame(corpus_df.iloc[0:0], "a")


def test_validate_corpus_frame_rejects_all_null_text(corpus_df):
    broken = corpus_df.copy()
    broken["text"] = None
    with pytest.raises(DataError, match="entirely null"):
        validate_corpus_frame(broken, "a")


def test_validate_corpus_frame_keeps_created_at_utc(corpus_df):
    out = validate_corpus_frame(corpus_df, "a")
    assert out["created_at"].dt.tz is not None


def test_validate_corpus_frame_rejects_naive_created_at(corpus_df):
    broken = corpus_df.copy()
    broken["created_at"] = pd.to_datetime(
        ["2024-01-01 09:00"] * len(broken)
    ) + pd.to_timedelta(range(len(broken)), unit="D")
    assert broken["created_at"].dt.tz is None
    with pytest.raises(DataError, match="not timezone-aware"):
        validate_corpus_frame(broken, "a")


def test_validate_corpus_frame_rejects_malformed_created_at(corpus_df):
    broken = corpus_df.copy()
    broken["created_at"] = broken["created_at"].astype(object)
    broken.loc[broken.index[0], "created_at"] = "not-a-timestamp"
    with pytest.raises(DataError, match="invalid created_at"):
        validate_corpus_frame(broken, "a")


# --------------------------------------------------------------------------
# loading / paths
# --------------------------------------------------------------------------


def test_load_corpus_rejects_unknown_label(tmp_path):
    with pytest.raises(DataError, match="unknown corpus label"):
        load_corpus("z", source_dir=tmp_path)


def test_load_corpus_missing_file(tmp_path):
    with pytest.raises(DataError, match="not found"):
        load_corpus("a", source_dir=tmp_path)


def test_load_corpus_round_trips_parquet(tmp_path, corpus_df):
    corpus_df.to_parquet(tmp_path / "corpus_a.parquet", index=False)
    out = load_corpus("a", source_dir=tmp_path)
    # The point of the whole module: after a parquet round-trip tokens are lists.
    assert isinstance(out.at[0, "tokens"], list)


def test_resolve_paths_creates_directories(tmp_path):
    paths = resolve_paths(tmp_path / "analysis", tmp_path / "outputs")
    assert isinstance(paths, Paths)
    assert paths.analysis.is_dir() and paths.outputs.is_dir()
