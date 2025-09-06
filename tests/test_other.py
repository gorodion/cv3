"""Tests for other functions in cv3.

This module contains tests for various utility functions provided by the cv3 library
that don't fit into the other specific categories. This includes color space conversion
functions and other miscellaneous utilities.

The tests verify that cv3 functions produce the same results as their native OpenCV
counterparts, ensuring compatibility and correctness.
"""

import numpy as np
import cv2
import cv3
import pytest

TEST_IMG = 'img.jpeg'
img_bgr = cv2.imread(TEST_IMG)
img_gray = cv2.imread(TEST_IMG, 0)
img = cv2.cvtColor(img_bgr, code=cv2.COLOR_RGB2BGR)


def test_gray2rgb():
    # GRAY to RGB
    rgb = cv3.gray2rgb(img_gray)
    assert rgb.shape == (*img_gray.shape, 3)

    # image with shape (height, width, 1)
    img_1 = img_gray[..., None]
    cv3.gray2rgb(img_1)
    assert rgb.shape == (*img_gray.shape, 3)

    # to RGBA
    rgba = cv3.gray2rgba(img_1)
    assert rgba.shape == (*img_gray.shape, 4)

    # image with shape (height, width, 3)
    with pytest.raises(ValueError):
        cv3.gray2rgba(img)
