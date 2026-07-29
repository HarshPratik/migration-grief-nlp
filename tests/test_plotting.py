"""Smoke tests for scripts/plotting.py.

Figures are checked by eye, not by assertion, so this file only proves the
module imports under Agg, produces a real file, and still validates the one
thing a reader cannot spot in a rendered PNG: mislabelled axes.
"""

from __future__ import annotations

import numpy as np
import pytest

from plotting import apply_style, fig_heatmap, save_fig


def test_apply_style_returns_a_style_dict_with_dpi():
    style = apply_style()
    assert isinstance(style, dict)
    assert style["dpi"] == 300
    assert apply_style({"dpi": 150})["dpi"] == 150


def test_fig_heatmap_and_save_fig_write_a_png(tmp_path, style):
    matrix = np.arange(6, dtype=float).reshape(2, 3)
    fig, _ = fig_heatmap(matrix, ["r0", "r1"], ["c0", "c1", "c2"], style=style, title="t")
    (written,) = save_fig(fig, tmp_path / "heat", dpi=72)
    assert written.suffix == ".png"
    assert written.stat().st_size > 0


def test_fig_heatmap_rejects_label_shape_mismatch(style):
    with pytest.raises(ValueError, match="does not match labels"):
        fig_heatmap(np.zeros((2, 3)), ["r0", "r1"], ["c0", "c1"], style=style)
