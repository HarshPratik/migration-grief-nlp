"""Tests for scripts/text_windows.py.

Window boundaries are the #1 correctness risk in the project: every emotion
number in Phase 3 is computed over these spans, so an off-by-one at a document
edge or a negator amputated by sentence clipping changes a reported result
rather than crashing. Each boundary rule therefore gets its own test.
"""

from __future__ import annotations

import pytest

from text_windows import (
    NEGATORS,
    SurfaceMatcher,
    Window,
    build_token_index,
    collapse_phrases,
    dedupe_overlapping,
    doc_freq,
    extract_windows,
    find_anchor_positions,
    kwic_line,
    simple_tokens,
    vocab_counts,
)

# 0    1    2    3    4    5         6
TOKENS = ["a", "b", "c", "d", "e", "ANCHOR", "f"]


# --------------------------------------------------------------------------
# extract_windows - boundaries
# --------------------------------------------------------------------------


def test_extract_windows_mid_document():
    tokens = ["w0", "w1", "w2", "ANCHOR", "w4", "w5", "w6"]
    (window,) = extract_windows(tokens, {"ANCHOR": [3]}, size=2)
    assert (window.start, window.position, window.end) == (1, 3, 5)
    assert window.tokens(tokens) == ["w1", "w2", "ANCHOR", "w4", "w5"]
    assert not window.truncated_left and not window.truncated_right


def test_extract_windows_truncates_at_document_start():
    tokens = ["ANCHOR", "b", "c", "d"]
    (window,) = extract_windows(tokens, {"ANCHOR": [0]}, size=2)
    assert (window.start, window.end) == (0, 2)
    assert window.truncated_left is True
    assert window.truncated_right is False
    assert window.n_left == 0 and window.n_right == 2


def test_extract_windows_truncates_at_document_end():
    tokens = ["a", "b", "ANCHOR"]
    (window,) = extract_windows(tokens, {"ANCHOR": [2]}, size=2)
    assert (window.start, window.end) == (0, 2)
    assert window.truncated_left is False
    assert window.truncated_right is True


def test_extract_windows_larger_than_document():
    tokens = ["a", "ANCHOR", "c"]
    (window,) = extract_windows(tokens, {"ANCHOR": [1]}, size=50)
    assert (window.start, window.end) == (0, 2)
    assert window.truncated_left and window.truncated_right


def test_extract_windows_anchor_is_sole_token():
    (window,) = extract_windows(["ANCHOR"], {"ANCHOR": [0]}, size=5)
    assert (window.start, window.end) == (0, 0)
    assert window.tokens(["ANCHOR"]) == ["ANCHOR"]
    assert window.n_left == 0 and window.n_right == 0


def test_extract_windows_two_adjacent_anchors():
    tokens = ["x", "A", "B", "y"]
    windows = extract_windows(tokens, {"A": [1], "B": [2]}, size=1)
    assert [w.anchor for w in windows] == ["A", "B"]  # ordered by anchor
    assert [(w.start, w.end) for w in windows] == [(0, 2), (1, 3)]


def test_extract_windows_empty_document():
    assert extract_windows([], {"ANCHOR": [0]}, size=5) == []


def test_extract_windows_size_zero_keeps_only_the_anchor():
    (window,) = extract_windows(TOKENS, {"ANCHOR": [5]}, size=0)
    assert (window.start, window.end) == (5, 5)


def test_extract_windows_ignores_out_of_range_positions():
    assert extract_windows(TOKENS, {"ANCHOR": [99]}, size=2) == []


def test_extract_windows_invalid_size_raises():
    with pytest.raises(ValueError, match="window size must be >= 0"):
        extract_windows(TOKENS, {"ANCHOR": [5]}, size=-1)


def test_extract_windows_mismatched_sent_idx_raises():
    with pytest.raises(ValueError, match="does not match tokens length"):
        extract_windows(TOKENS, {"ANCHOR": [5]}, size=2, sent_idx=[0, 0, 0])


def test_extract_windows_max_per_anchor_cap():
    tokens = ["ANCHOR", "x", "ANCHOR", "x", "ANCHOR"]
    windows = extract_windows(tokens, {"ANCHOR": [0, 2, 4]}, size=1, max_per_anchor=2)
    assert [w.position for w in windows] == [0, 2]


def test_extract_windows_never_mutates_inputs():
    tokens = list(TOKENS)
    positions = {"ANCHOR": [5]}
    extract_windows(tokens, positions, size=3, negation_rescue=2)
    assert tokens == TOKENS
    assert positions == {"ANCHOR": [5]}


# --------------------------------------------------------------------------
# extract_windows - sentence clipping
# --------------------------------------------------------------------------

# indices:  0    1    2    3     4    5    6         7    8     9    10
SENT_TOKENS = ["a", "b", "c", ".", "d", "e", "ANCHOR", "f", ".", "g", "h"]
SENT_IDX = [0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2]


def test_extract_windows_clips_at_sentence_boundary():
    (window,) = extract_windows(
        SENT_TOKENS, {"ANCHOR": [6]}, size=5, sent_idx=SENT_IDX, clip_at_sentence=True
    )
    assert (window.start, window.end) == (4, 8)
    assert window.tokens(SENT_TOKENS) == ["d", "e", "ANCHOR", "f", "."]
    assert window.sent_idx == 1


def test_extract_windows_without_clipping_spans_sentences():
    (window,) = extract_windows(
        SENT_TOKENS, {"ANCHOR": [6]}, size=5, sent_idx=SENT_IDX, clip_at_sentence=False
    )
    assert (window.start, window.end) == (1, 10)
    assert "a" not in window.tokens(SENT_TOKENS)  # size clamp still applies
    assert "g" in window.tokens(SENT_TOKENS)      # but the sentence edge does not


def test_extract_windows_records_sentence_index():
    (window,) = extract_windows(SENT_TOKENS, {"ANCHOR": [6]}, size=1, sent_idx=SENT_IDX)
    assert window.sent_idx == 1


def test_extract_windows_sent_idx_none_leaves_field_none():
    (window,) = extract_windows(TOKENS, {"ANCHOR": [5]}, size=1)
    assert window.sent_idx is None


# --------------------------------------------------------------------------
# extract_windows - negation rescue
# --------------------------------------------------------------------------

RESCUE_TOKENS = ["i", "do", "not", "really", "ever", "ANCHOR", "home"]


def test_negation_rescue_pulls_negator_back_in():
    (window,) = extract_windows(RESCUE_TOKENS, {"ANCHOR": [5]}, size=2, negation_rescue=4)
    assert window.negation_rescued is True
    assert window.start == 2
    assert "not" in window.tokens(RESCUE_TOKENS)


def test_negation_rescue_off_amputates_the_negator():
    (window,) = extract_windows(RESCUE_TOKENS, {"ANCHOR": [5]}, size=2, negation_rescue=0)
    assert window.negation_rescued is False
    assert "not" not in window.tokens(RESCUE_TOKENS)


def test_negation_rescue_does_not_cross_a_sentence_boundary():
    tokens = ["not", "x", ".", "y", "ANCHOR", "z"]
    sent_idx = [0, 0, 0, 1, 1, 1]
    (window,) = extract_windows(
        tokens, {"ANCHOR": [4]}, size=1, sent_idx=sent_idx,
        clip_at_sentence=True, negation_rescue=4,
    )
    assert window.negation_rescued is False
    assert window.start == 3
    assert "not" not in window.tokens(tokens)


def test_negation_rescue_reaches_across_when_no_sentence_info():
    # Same tokens, same distance - only the sentence information differs, which
    # isolates the boundary rule from the distance rule.
    tokens = ["not", "x", ".", "y", "ANCHOR", "z"]
    (window,) = extract_windows(tokens, {"ANCHOR": [4]}, size=1, negation_rescue=4)
    assert window.negation_rescued is True
    assert window.start == 0


def test_negation_rescue_respects_its_distance_limit():
    tokens = ["not", "a", "b", "c", "d", "e", "ANCHOR"]
    (window,) = extract_windows(tokens, {"ANCHOR": [6]}, size=1, negation_rescue=2)
    assert window.negation_rescued is False
    assert window.start == 5


def test_negators_include_the_contraction_forms():
    for token in ("not", "never", "n't", "cannot", "without"):
        assert token in NEGATORS


# --------------------------------------------------------------------------
# dedupe_overlapping / kwic_line
# --------------------------------------------------------------------------


def test_dedupe_overlapping_drops_covered_repeat():
    tokens = ["A", "x", "A", "y", "z"]
    windows = extract_windows(tokens, {"A": [0, 2]}, size=2)
    assert len(windows) == 2
    kept = dedupe_overlapping(windows)
    assert [w.position for w in kept] == [0]


def test_dedupe_overlapping_keeps_distant_repeats():
    tokens = ["A"] + ["x"] * 10 + ["A"]
    windows = extract_windows(tokens, {"A": [0, 11]}, size=2)
    assert [w.position for w in dedupe_overlapping(windows)] == [0, 11]


def test_dedupe_overlapping_is_per_anchor():
    tokens = ["A", "B", "x"]
    windows = extract_windows(tokens, {"A": [0], "B": [1]}, size=2)
    assert len(dedupe_overlapping(windows)) == 2


def test_dedupe_overlapping_does_not_mutate_input():
    windows = extract_windows(["A", "x", "A"], {"A": [0, 2]}, size=2)
    original = list(windows)
    dedupe_overlapping(windows)
    assert windows == original


def test_kwic_line_format():
    tokens = ["p", "q", "ANCHOR", "r", "s"]
    window = Window("ANCHOR", 2, 0, 4, None, False, False, False)
    assert kwic_line(tokens, window) == "p q | ANCHOR | r s"


def test_kwic_line_empty_left_context():
    tokens = ["ANCHOR", "r"]
    window = Window("ANCHOR", 0, 0, 1, None, True, False, False)
    assert kwic_line(tokens, window) == " | ANCHOR | r"


# --------------------------------------------------------------------------
# indexing
# --------------------------------------------------------------------------


def test_build_token_index_ascending_positions():
    assert build_token_index(["a", "b", "a", "c", "a"]) == {"a": [0, 2, 4], "b": [1], "c": [3]}


def test_build_token_index_empty():
    assert build_token_index([]) == {}


def test_find_anchor_positions_omits_absent_terms():
    found = find_anchor_positions(["a", "b", "a"], ["a", "zzz"])
    assert found == {"a": [0, 2]}


def test_find_anchor_positions_accepts_prebuilt_index():
    index = build_token_index(["a", "b", "a"])
    assert find_anchor_positions([], ["a"], index=index) == {"a": [0, 2]}


def test_find_anchor_positions_copies_the_index_lists():
    index = {"a": [0, 2]}
    found = find_anchor_positions([], ["a"], index=index)
    found["a"].append(99)
    assert index["a"] == [0, 2]


def test_simple_tokens_lowercases_and_strips_punctuation():
    assert simple_tokens("I don't Miss Home!") == ["i", "don't", "miss", "home"]


def test_simple_tokens_handles_none():
    assert simple_tokens(None) == []


def test_vocab_counts_and_doc_freq_differ():
    documents = [["a", "a", "b"], ["a"]]
    assert vocab_counts(documents)["a"] == 3
    assert doc_freq(documents)["a"] == 2


# --------------------------------------------------------------------------
# collapse_phrases
# --------------------------------------------------------------------------


def test_collapse_phrases_longest_match_wins():
    tokens = ["i", "had", "a", "video", "call", "today"]
    surface = {"video call": "video_call", "call": "call"}
    assert collapse_phrases(tokens, surface) == ["i", "had", "a", "video_call", "today"]


def test_collapse_phrases_lowercases_passthrough_tokens():
    assert collapse_phrases(["Home", "SICK"], {}) == ["Home", "SICK"]
    assert collapse_phrases(["Home", "SICK"], {"home": "home"}) == ["home", "sick"]


def test_collapse_phrases_empty_inputs():
    assert collapse_phrases([], {"a b": "ab"}) == []


def test_collapse_phrases_does_not_mutate_input():
    tokens = ["video", "call"]
    collapse_phrases(tokens, {"video call": "video_call"})
    assert tokens == ["video", "call"]


def test_collapse_phrases_max_span_limits_the_scan():
    tokens = ["a", "b", "c"]
    surface = {"a b c": "abc"}
    assert collapse_phrases(tokens, surface, max_span=2) == ["a", "b", "c"]
    assert collapse_phrases(tokens, surface) == ["abc"]


# --------------------------------------------------------------------------
# SurfaceMatcher
# --------------------------------------------------------------------------

SURFACE_MAP = {
    "wi-fi": "wifi",
    "video call": "video_call",
    "away": "away",
    "timezone": "timezone",
}


def test_surface_matcher_splits_single_and_multi():
    matcher = SurfaceMatcher.from_surface_map(SURFACE_MAP)
    assert set(matcher.single) == {"away", "timezone"}
    assert set(matcher.multi) == {"wifi", "videocall"}


def test_surface_matcher_collapses_hyphen_split_tokens():
    matcher = SurfaceMatcher.from_surface_map(SURFACE_MAP)
    assert matcher.collapse(["wi", "-", "fi"]) == [("wifi", 0, 3)]


def test_surface_matcher_collapses_multiword():
    matcher = SurfaceMatcher.from_surface_map(SURFACE_MAP)
    assert matcher.collapse(["video", "call"]) == [("video_call", 0, 2)]


def test_surface_matcher_does_not_join_single_word_forms_across_tokens():
    # "in a way" concatenates to "inaway"; "a"+"way" concatenates to "away".
    # A single-word surface form must never match across a token boundary or
    # the anchor "away" swallows this phrase.
    matcher = SurfaceMatcher.from_surface_map(SURFACE_MAP)
    assert matcher.collapse(["in", "a", "way"]) == [("in", 0, 1), ("a", 1, 1), ("way", 2, 1)]


def test_surface_matcher_maps_single_form():
    matcher = SurfaceMatcher.from_surface_map(SURFACE_MAP)
    assert matcher.collapse(["Away"]) == [("away", 0, 1)]


def test_surface_matcher_passes_through_unknown_tokens_lowercased():
    matcher = SurfaceMatcher.from_surface_map(SURFACE_MAP)
    assert matcher.collapse(["Hello"]) == [("hello", 0, 1)]


def test_surface_matcher_spans_cover_the_input_exactly():
    matcher = SurfaceMatcher.from_surface_map(SURFACE_MAP)
    tokens = ["i", "had", "a", "video", "call", "over", "wi", "-", "fi"]
    emitted = matcher.collapse(tokens)
    assert sum(span for _, _, span in emitted) == len(tokens)
    assert [term for term, _, _ in emitted][3] == "video_call"


def test_surface_matcher_does_not_mutate_input():
    matcher = SurfaceMatcher.from_surface_map(SURFACE_MAP)
    tokens = ["video", "call"]
    matcher.collapse(tokens)
    assert tokens == ["video", "call"]
