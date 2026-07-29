"""Unit tests for scripts/contextual_vectors.py (no BERT / no network)."""

from __future__ import annotations

import numpy as np
import pytest

from contextual_vectors import (
    all_but_the_top,
    average_profiles,
    neighbors_from_sense,
    type_embeddings,
)


# Three docs, dim=2. Tokens: cat, dog, bird.
# doc0: cat=[1,0], dog=[0,1]
# doc1: cat=[3,0], bird=[0,2]
# doc2: dog=[0,3], bird=[1,1]
DOC_IDX = np.array([0, 0, 1, 1, 2, 2], dtype=np.int32)
TOKEN_STR = np.array(["cat", "dog", "cat", "bird", "dog", "bird"], dtype=object)
VECTORS = np.array(
    [
        [1.0, 0.0],
        [0.0, 1.0],
        [3.0, 0.0],
        [0.0, 2.0],
        [0.0, 3.0],
        [1.0, 1.0],
    ],
    dtype=np.float32,
)


def test_average_profiles_full_corpus():
    targets = {
        "feline": ("cat",),
        "canine": ("dog",),
        "pets": ("cat", "dog"),
    }
    profiles = average_profiles(DOC_IDX, TOKEN_STR, VECTORS, targets)
    # cat rows: [1,0] and [3,0] -> mean [2, 0]
    np.testing.assert_allclose(profiles["feline"], [2.0, 0.0])
    # dog rows: [0,1] and [0,3] -> mean [0, 2]
    np.testing.assert_allclose(profiles["canine"], [0.0, 2.0])
    # pets: all four cat+dog vectors -> mean [1, 1]
    np.testing.assert_allclose(profiles["pets"], [1.0, 1.0])


def test_average_profiles_with_doc_mask():
    targets = {"feline": ("cat",)}
    mask = np.array([True, False, False], dtype=bool)  # only doc 0
    profiles = average_profiles(DOC_IDX, TOKEN_STR, VECTORS, targets, mask)
    np.testing.assert_allclose(profiles["feline"], [1.0, 0.0])


def test_average_profiles_empty_target_is_zero():
    targets = {"missing": ("zebra",)}
    profiles = average_profiles(DOC_IDX, TOKEN_STR, VECTORS, targets)
    np.testing.assert_allclose(profiles["missing"], [0.0, 0.0])


def test_type_embeddings_present_types_only():
    names, matrix = type_embeddings(
        DOC_IDX, TOKEN_STR, VECTORS, vocabulary=("cat", "dog", "zebra")
    )
    assert names == ("cat", "dog")
    assert "zebra" not in names
    # cat mean [2,0], dog mean [0,2]
    cat_i = names.index("cat")
    dog_i = names.index("dog")
    np.testing.assert_allclose(matrix[cat_i], [2.0, 0.0])
    np.testing.assert_allclose(matrix[dog_i], [0.0, 2.0])


def test_type_embeddings_respects_doc_mask():
    mask = np.array([False, True, False], dtype=bool)  # only doc 1
    names, matrix = type_embeddings(
        DOC_IDX, TOKEN_STR, VECTORS, vocabulary=("cat", "dog", "bird"), doc_mask=mask
    )
    assert set(names) == {"bird", "cat"}
    cat_i = names.index("cat")
    np.testing.assert_allclose(matrix[cat_i], [3.0, 0.0])


def test_neighbors_from_sense_orders_by_cosine():
    type_names = ("a", "b", "c")
    type_matrix = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=np.float64,
    )
    sense = np.array([1.0, 0.0])
    pairs = neighbors_from_sense(sense, type_names, type_matrix, k=2, exclude=())
    assert [w for w, _ in pairs] == ["a", "c"]
    assert pairs[0][1] == pytest.approx(1.0)


def test_neighbors_from_sense_excludes_self():
    type_names = ("cat", "dog")
    type_matrix = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    sense = np.array([1.0, 0.0])
    pairs = neighbors_from_sense(sense, type_names, type_matrix, k=5, exclude={"cat"})
    assert [w for w, _ in pairs] == ["dog"]


def test_neighbors_from_sense_empty_types():
    pairs = neighbors_from_sense(np.ones(2), (), np.zeros((0, 2)), k=5)
    assert pairs == []


def test_all_but_the_top_preserves_keys_and_reduces_rank():
    # Four nearly collinear profiles + one orthogonal component.
    profiles = {
        "w": np.array([1.0, 0.0, 0.0]),
        "x": np.array([1.1, 0.1, 0.0]),
        "y": np.array([0.9, -0.1, 0.0]),
        "z": np.array([1.0, 0.0, 1.0]),
    }
    corrected = all_but_the_top(profiles, n_components=1)
    assert set(corrected) == set(profiles)
    # Resulting matrix should have zero mean after correction path.
    matrix = np.vstack([corrected[n] for n in sorted(corrected)])
    assert matrix.shape == (4, 3)


def test_all_but_the_top_zero_components_is_copy():
    profiles = {"a": np.array([1.0, 2.0]), "b": np.array([3.0, 4.0])}
    out = all_but_the_top(profiles, 0)
    np.testing.assert_allclose(out["a"], [1.0, 2.0])
    assert out["a"] is not profiles["a"]
