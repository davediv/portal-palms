"""The mirror transform.

The spec calls this out specifically, and rightly: the preview is displayed
mirrored, so every consumer downstream of detection must see mirrored
coordinates. Get it wrong and the app is not broken in any obvious way -- it
just feels subtly, unplaceably off, and the cause is very hard to find later.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from portal_palms import config as C
from portal_palms.tracking import mirror_landmarks

Arr = NDArray[np.float32]


def make_landmarks() -> Arr:
    rng = np.random.default_rng(11)
    return rng.random((C.NUM_HANDS, C.NUM_LANDMARKS, 3), dtype=np.float32)


def test_x_is_reflected_about_the_centre() -> None:
    lm = make_landmarks()
    original = lm.copy()
    mirror_landmarks(lm)
    np.testing.assert_allclose(lm[..., 0], 1.0 - original[..., 0], rtol=0, atol=1e-6)


def test_y_and_z_are_untouched() -> None:
    lm = make_landmarks()
    original = lm.copy()
    mirror_landmarks(lm)
    np.testing.assert_array_equal(lm[..., 1], original[..., 1])
    np.testing.assert_array_equal(lm[..., 2], original[..., 2])


def test_mirroring_twice_is_the_identity() -> None:
    lm = make_landmarks()
    original = lm.copy()
    mirror_landmarks(mirror_landmarks(lm))
    np.testing.assert_allclose(lm, original, rtol=0, atol=1e-6)


def test_mutates_in_place_and_returns_the_same_array() -> None:
    lm = make_landmarks()
    out = mirror_landmarks(lm)
    assert out is lm, "must not allocate a new array: this runs per detection"


def test_a_hand_on_the_left_ends_up_on_the_right() -> None:
    lm = np.zeros((C.NUM_HANDS, C.NUM_LANDMARKS, 3), dtype=np.float32)
    lm[0, C.WRIST, 0] = 0.15  # near the left edge of the raw camera frame
    lm[1, C.WRIST, 0] = 0.80
    mirror_landmarks(lm)
    assert lm[0, C.WRIST, 0] == pytest.approx(0.85, abs=1e-6)
    assert lm[1, C.WRIST, 0] == pytest.approx(0.20, abs=1e-6)


def test_centre_is_a_fixed_point() -> None:
    lm = np.full((C.NUM_HANDS, C.NUM_LANDMARKS, 3), 0.5, dtype=np.float32)
    mirror_landmarks(lm)
    np.testing.assert_allclose(lm[..., 0], 0.5, atol=1e-7)


def test_dtype_is_preserved() -> None:
    """float64 would silently double every upload and mismatch the GL format."""
    lm = make_landmarks()
    mirror_landmarks(lm)
    assert lm.dtype == np.float32
