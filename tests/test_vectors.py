"""Tests for scripts/vectors.py.

These functions produce the numbers that go in the paper, so the tests check
arithmetic against hand-computed values rather than against the implementation's
own output, and pin the two properties the module promises: determinism (tie
breaks) and honesty (chance baselines, normalised RBO, attenuation caps).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import scipy.sparse as sp

from vectors import (
    Reliability,
    attenuation_correct,
    build_cooccurrence,
    build_vocabulary,
    codocument_counts,
    cosine,
    cosine_matrix,
    jaccard_at_k,
    jaccard_chance,
    l2_normalize,
    ppmi,
    rank_biased_overlap,
    reliability_ceiling,
    spearman_brown,
    split_half_reliability,
    top_k_neighbors,
    umass_coherence,
)

# A symmetric 3x3 toy with a total of 12, chosen so every marginal is a clean
# fraction and each PPMI cell can be written out by hand.
TOY = np.array([[0.0, 2.0, 1.0], [2.0, 0.0, 1.0], [1.0, 1.0, 4.0]])


# --------------------------------------------------------------------------
# ppmi
# --------------------------------------------------------------------------


def test_ppmi_cell_matches_explicit_arithmetic():
    result = ppmi(sp.csr_matrix(TOY)).toarray()
    # total = 12; row_sum[0] = 3; col_sum[1] = 3
    #   p(0,1) = 2/12, p(0) = 3/12, p_ctx(1) = 3/12
    expected = math.log((2 / 12) / ((3 / 12) * (3 / 12)))
    assert result[0, 1] == pytest.approx(expected)
    assert result[0, 1] == pytest.approx(0.9808292530, abs=1e-9)


def test_ppmi_diagonal_cell_matches_explicit_arithmetic():
    result = ppmi(sp.csr_matrix(TOY)).toarray()
    expected = math.log((4 / 12) / ((6 / 12) * (6 / 12)))
    assert result[2, 2] == pytest.approx(expected)


def test_ppmi_clips_negative_pmi_to_zero():
    result = ppmi(sp.csr_matrix(TOY)).toarray()
    # pmi(0,2) = log((1/12) / (0.25 * 0.5)) = log(2/3) < 0
    assert result[0, 2] == 0.0


def test_ppmi_has_no_negative_entries():
    result = ppmi(sp.csr_matrix(TOY))
    assert result.data.size == 0 or result.data.min() >= 0.0


def test_ppmi_is_symmetric_for_symmetric_input():
    result = ppmi(sp.csr_matrix(TOY)).toarray()
    assert np.abs(result - result.T).max() == pytest.approx(0.0)


def test_ppmi_shift_k_reduces_values():
    plain = ppmi(sp.csr_matrix(TOY)).toarray()
    shifted = ppmi(sp.csr_matrix(TOY), shift_k=2).toarray()
    assert (shifted <= plain + 1e-12).all()
    assert shifted.max() < plain.max()
    assert shifted[0, 1] == pytest.approx(plain[0, 1] - math.log(2))


def test_ppmi_cds_smoothing_runs_and_changes_values():
    smoothed = ppmi(sp.csr_matrix(TOY), cds_smoothing=0.75).toarray()
    plain = ppmi(sp.csr_matrix(TOY)).toarray()
    assert smoothed.shape == plain.shape
    assert not np.allclose(smoothed, plain)


def test_ppmi_of_empty_matrix_is_empty():
    result = ppmi(sp.csr_matrix((3, 3)))
    assert result.shape == (3, 3)
    assert result.nnz == 0


def test_ppmi_positive_false_keeps_negative_pmi_out_of_the_kept_mask():
    # `positive=False` skips the clip, but the sparse result still only stores
    # strictly-positive values, so the cell drops out either way.
    result = ppmi(sp.csr_matrix(TOY), positive=False).toarray()
    assert result[0, 2] == 0.0
    assert result[0, 1] > 0.0


# --------------------------------------------------------------------------
# build_vocabulary / build_cooccurrence
# --------------------------------------------------------------------------


def test_build_vocabulary_is_sorted_and_filtered():
    documents = [["b", "a", "a"], ["a", "c"]]
    assert build_vocabulary(documents, min_count=2) == ("a",)
    assert build_vocabulary(documents) == ("a", "b", "c")


def test_build_vocabulary_max_size_keeps_the_most_frequent():
    documents = [["a"] * 5 + ["b"] * 3 + ["c"]]
    assert build_vocabulary(documents, max_size=2) == ("a", "b")


def test_build_cooccurrence_never_crosses_document_boundaries():
    vocabulary = ("a", "b", "c", "d")
    matrix = build_cooccurrence([["a", "b"], ["c", "d"]], vocabulary, window=5).toarray()
    b, c = vocabulary.index("b"), vocabulary.index("c")
    assert matrix[b, c] == 0.0, "counted across a document boundary"
    assert matrix[c, b] == 0.0
    assert matrix[vocabulary.index("a"), b] == 1.0


def test_build_cooccurrence_respects_the_window():
    vocabulary = ("a", "b", "x")
    document = [["a", "x", "x", "x", "x", "x", "b"]]
    near = build_cooccurrence(document, vocabulary, window=2).toarray()
    far = build_cooccurrence(document, vocabulary, window=6).toarray()
    assert near[0, 1] == 0.0
    assert far[0, 1] == 1.0


def test_build_cooccurrence_is_symmetric_when_asked():
    vocabulary = ("a", "b", "c")
    matrix = build_cooccurrence([["a", "b", "c"]], vocabulary, window=2).toarray()
    assert np.array_equal(matrix, matrix.T)


def test_build_cooccurrence_asymmetric_mode_counts_forward_only():
    vocabulary = ("a", "b")
    matrix = build_cooccurrence([["a", "b"]], vocabulary, window=2, symmetric=False).toarray()
    assert matrix[0, 1] == 1.0
    assert matrix[1, 0] == 0.0


def test_build_cooccurrence_ignores_out_of_vocabulary_tokens():
    matrix = build_cooccurrence([["a", "zzz", "b"]], ("a", "b"), window=5).toarray()
    assert matrix[0, 1] == 1.0
    assert matrix.sum() == 2.0


def test_build_cooccurrence_rejects_window_below_one():
    with pytest.raises(ValueError, match="window must be >= 1"):
        build_cooccurrence([["a"]], ("a",), window=0)


# --------------------------------------------------------------------------
# similarity
# --------------------------------------------------------------------------


def test_cosine_identical_vectors():
    v = np.array([1.0, 2.0, 3.0])
    assert cosine(v, v) == pytest.approx(1.0)


def test_cosine_orthogonal_vectors():
    assert cosine(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(0.0)


def test_cosine_zero_vector_returns_zero_not_nan():
    value = cosine(np.zeros(3), np.ones(3))
    assert value == 0.0
    assert not math.isnan(value)


def test_cosine_is_scale_invariant():
    a = np.array([1.0, 2.0, 3.0])
    assert cosine(a, a * 7.0) == pytest.approx(1.0)


def test_cosine_opposite_vectors():
    assert cosine(np.array([1.0, 0.0]), np.array([-1.0, 0.0])) == pytest.approx(-1.0)


def test_cosine_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape mismatch"):
        cosine(np.ones(3), np.ones(4))


def test_l2_normalize_leaves_zero_rows_as_zero():
    out = l2_normalize(np.array([[3.0, 4.0], [0.0, 0.0]]))
    assert out[0].tolist() == pytest.approx([0.6, 0.8])
    assert out[1].tolist() == [0.0, 0.0]
    assert not np.isnan(out).any()


def test_cosine_matrix_diagonal_is_one():
    matrix = np.array([[1.0, 0.0], [1.0, 1.0]])
    assert np.diag(cosine_matrix(matrix)).tolist() == pytest.approx([1.0, 1.0])


# --------------------------------------------------------------------------
# top_k_neighbors
# --------------------------------------------------------------------------


def test_top_k_neighbors_orders_by_descending_similarity():
    result = top_k_neighbors([0.1, 0.9, 0.5], ["low", "high", "mid"], k=3)
    assert [w for w, _ in result] == ["high", "mid", "low"]


def test_top_k_neighbors_breaks_ties_alphabetically():
    result = top_k_neighbors([0.5, 0.5, 0.5], ["zeta", "alpha", "mid"], k=3)
    assert [w for w, _ in result] == ["alpha", "mid", "zeta"]


def test_top_k_neighbors_is_deterministic_across_calls():
    scores = [0.5] * 6
    vocabulary = ["f", "e", "d", "c", "b", "a"]
    first = top_k_neighbors(scores, vocabulary, k=3)
    second = top_k_neighbors(scores, vocabulary, k=3)
    assert first == second == [("a", 0.5), ("b", 0.5), ("c", 0.5)]


def test_top_k_neighbors_excludes_self():
    result = top_k_neighbors([1.0, 0.4], ["home", "house"], k=2, exclude=["home"])
    assert [w for w, _ in result] == ["house"]


def test_top_k_neighbors_exclude_is_case_insensitive():
    result = top_k_neighbors([1.0, 0.4], ["Home", "house"], k=2, exclude=["HOME"])
    assert [w for w, _ in result] == ["house"]


def test_top_k_neighbors_k_larger_than_vocabulary():
    result = top_k_neighbors([0.2, 0.3], ["a", "b"], k=15)
    assert len(result) == 2


def test_top_k_neighbors_applies_min_similarity():
    result = top_k_neighbors([0.9, 0.1], ["a", "b"], k=5, min_similarity=0.5)
    assert [w for w, _ in result] == ["a"]


def test_top_k_neighbors_drops_nan_scores():
    result = top_k_neighbors([float("nan"), 0.3], ["a", "b"], k=5)
    assert [w for w, _ in result] == ["b"]


def test_top_k_neighbors_length_mismatch_raises():
    with pytest.raises(ValueError, match="differ in length"):
        top_k_neighbors([0.1, 0.2], ["a", "b", "c"])


# --------------------------------------------------------------------------
# neighbourhood comparison
# --------------------------------------------------------------------------


def test_jaccard_at_k_identical_lists():
    assert jaccard_at_k(["a", "b", "c"], ["a", "b", "c"]) == 1.0


def test_jaccard_at_k_disjoint_lists():
    assert jaccard_at_k(["a", "b"], ["c", "d"]) == 0.0


def test_jaccard_at_k_partial_overlap():
    assert jaccard_at_k(["a", "b"], ["b", "c"]) == pytest.approx(1 / 3)


def test_jaccard_at_k_empty_lists():
    assert jaccard_at_k([], []) == 0.0


def test_jaccard_at_k_truncates_at_k():
    assert jaccard_at_k(["a", "b", "z"], ["a", "b", "q"], k=2) == 1.0


def test_jaccard_chance_matches_the_reported_baseline():
    # k / (2V - k) = 15 / 841 - the value printed next to every Jaccard.
    assert jaccard_chance(15, 428) == pytest.approx(0.0178, abs=5e-5)
    assert jaccard_chance(15, 428) == pytest.approx(15 / 841)


def test_jaccard_chance_degenerate_inputs():
    assert jaccard_chance(0, 428) == 0.0
    assert jaccard_chance(15, 0) == 0.0


def test_rank_biased_overlap_identical_lists_is_exactly_one():
    ranking = list("abcde")
    assert rank_biased_overlap(ranking, ranking) == pytest.approx(1.0)


def test_rank_biased_overlap_disjoint_lists_is_zero():
    assert rank_biased_overlap(list("abc"), list("xyz")) == pytest.approx(0.0)


def test_rank_biased_overlap_penalises_a_top_swap_more_than_a_bottom_swap():
    ranking = list("abcde")
    top_swap = rank_biased_overlap(ranking, list("bacde"))
    bottom_swap = rank_biased_overlap(ranking, list("abced"))
    assert top_swap < bottom_swap < 1.0


def test_rank_biased_overlap_unnormalised_caps_below_one():
    ranking = list("abcdefghijklmno")  # depth 15
    assert rank_biased_overlap(ranking, ranking, normalize=False) == pytest.approx(
        1.0 - 0.9**15
    )


def test_rank_biased_overlap_empty_lists():
    assert rank_biased_overlap([], []) == 0.0


@pytest.mark.parametrize("p", [0.0, 1.0, -0.1, 1.5])
def test_rank_biased_overlap_rejects_p_outside_the_open_unit_interval(p):
    with pytest.raises(ValueError, match=r"p must be in \(0,1\)"):
        rank_biased_overlap(list("ab"), list("ab"), p=p)


# --------------------------------------------------------------------------
# reliability
# --------------------------------------------------------------------------


def test_spearman_brown_projects_to_full_length():
    assert spearman_brown(0.5) == pytest.approx(2 / 3)
    assert spearman_brown(1.0) == pytest.approx(1.0)


def test_spearman_brown_floors_at_zero():
    assert spearman_brown(-0.5) == 0.0
    assert spearman_brown(-1.0) == 0.0
    assert spearman_brown(0.0) == 0.0


def test_attenuation_correct_caps_at_one():
    assert attenuation_correct(0.9, 0.5, 0.5) == 1.0


def test_attenuation_correct_scales_by_the_ceiling():
    assert attenuation_correct(0.4, 0.64, 0.64) == pytest.approx(0.625)


def test_attenuation_correct_returns_nan_when_a_reliability_is_zero():
    assert math.isnan(attenuation_correct(0.5, 0.0, 0.8))
    assert math.isnan(attenuation_correct(0.5, 0.8, 0.0))


def test_reliability_ceiling_is_the_geometric_mean():
    assert reliability_ceiling(0.36, 0.64) == pytest.approx(0.48)


def test_split_half_reliability_with_a_deterministic_build_vector():
    documents = [["a"], ["b"], ["c"], ["d"], ["e"], ["f"]]
    result = split_half_reliability(
        lambda subset: np.array([1.0, 2.0, 3.0]), documents, "home", n_splits=5
    )
    assert isinstance(result, Reliability)
    assert result.target == "home"
    assert result.n_splits == 5
    assert result.mean_split_half == pytest.approx(1.0)
    assert result.spearman_brown == pytest.approx(1.0)
    assert result.std == pytest.approx(0.0)
    assert result.usable is True


def test_split_half_reliability_is_reproducible_for_a_fixed_seed():
    documents = [[str(i)] for i in range(10)]

    def build(subset):
        return np.array([float(len(subset)), 1.0, 0.0])

    first = split_half_reliability(
        build, documents, "t", n_splits=5, rng=np.random.default_rng(7)
    )
    second = split_half_reliability(
        build, documents, "t", n_splits=5, rng=np.random.default_rng(7)
    )
    assert first == second


def test_split_half_reliability_too_few_documents():
    result = split_half_reliability(lambda s: np.ones(2), [["a"], ["b"]], "t")
    assert result.n_splits == 0
    assert result.spearman_brown == 0.0
    assert result.usable is False


def test_split_half_reliability_when_the_vector_cannot_be_built():
    documents = [["a"], ["b"], ["c"], ["d"]]
    result = split_half_reliability(lambda subset: None, documents, "t", n_splits=3)
    assert result.n_splits == 0
    assert result.mean_split_half == 0.0


# --------------------------------------------------------------------------
# topic coherence
# --------------------------------------------------------------------------


def test_umass_coherence_matches_explicit_arithmetic():
    value = umass_coherence(["a", "b"], {"a": 5, "b": 4}, {("a", "b"): 3})
    assert value == pytest.approx(math.log((3 + 1) / 5))


def test_umass_coherence_finds_the_pair_key_in_either_order():
    forward = umass_coherence(["a", "b"], {"a": 5, "b": 4}, {("a", "b"): 3})
    reverse = umass_coherence(["a", "b"], {"a": 5, "b": 4}, {("b", "a"): 3})
    assert forward == pytest.approx(reverse)


def test_umass_coherence_needs_at_least_two_terms():
    assert umass_coherence(["a"], {"a": 5}, {}) == 0.0


def test_umass_coherence_skips_terms_with_no_document_frequency():
    assert umass_coherence(["a", "b"], {}, {("a", "b"): 3}) == 0.0


def test_umass_coherence_rewards_co_occurrence():
    df = {"a": 10, "b": 10}
    together = umass_coherence(["a", "b"], df, {("a", "b"): 10})
    apart = umass_coherence(["a", "b"], df, {("a", "b"): 0})
    assert together > apart


def test_codocument_counts_toy():
    documents = [["a", "b"], ["b", "c"], ["a", "b", "c"]]
    df, codf = codocument_counts(documents, ["a", "b", "c"])
    assert df == {"a": 2, "b": 3, "c": 2}
    assert codf == {("a", "b"): 2, ("b", "c"): 2, ("a", "c"): 1}


def test_codocument_counts_ignores_terms_outside_the_vocabulary():
    df, codf = codocument_counts([["a", "zzz"]], ["a"])
    assert df == {"a": 1}
    assert codf == {}


def test_codocument_counts_deduplicates_within_a_document():
    df, _ = codocument_counts([["a", "a", "a"]], ["a"])
    assert df == {"a": 1}
