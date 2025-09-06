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


def test_opt_video():
    # Test video function with fps parameter
    original_fps = cv3.opt.FPS
    cv3.opt.video(fps=60)
    assert cv3.opt.FPS == 60
    
    # Test video function with fourcc parameter as string
    original_fourcc = cv3.opt.FOURCC
    cv3.opt.video(fourcc='MJPG')
    assert isinstance(cv3.opt.FOURCC, int)
    
    # Test video function with fourcc parameter as int
    fourcc_int = cv2.VideoWriter_fourcc(*'XVID')
    cv3.opt.video(fourcc=fourcc_int)
    assert cv3.opt.FOURCC == fourcc_int
    
    # Test assertions
    with pytest.raises(AssertionError):
        cv3.opt.video(fps=0)
    
    with pytest.raises(AssertionError):
        cv3.opt.video(fourcc='INVALID')
    
    # Restore original values
    cv3.opt.FPS = original_fps
    cv3.opt.FOURCC = original_fourcc


def test_opt_draw():
    # Test draw function with various parameters
    original_thickness = cv3.opt.THICKNESS
    original_color = cv3.opt.COLOR
    original_font = cv3.opt.FONT
    original_pt_radius = cv3.opt.PT_RADIUS
    original_scale = cv3.opt.SCALE
    original_line_type = cv3.opt.LINE_TYPE
    
    # Test thickness parameter
    cv3.opt.draw(thickness=3)
    assert cv3.opt.THICKNESS == 3
    
    # Test color parameter
    cv3.opt.draw(color='red')
    assert cv3.opt.COLOR == 'red'
    
    # Test font parameter
    cv3.opt.draw(font=cv2.FONT_HERSHEY_PLAIN)
    assert cv3.opt.FONT == cv2.FONT_HERSHEY_PLAIN
    
    # Test pt_radius parameter
    cv3.opt.draw(pt_radius=5)
    assert cv3.opt.PT_RADIUS == 5
    
    # Test scale parameter
    cv3.opt.draw(scale=2.0)
    assert cv3.opt.SCALE == 2.0
    
    # Test line_type parameter
    cv3.opt.draw(line_type=cv2.LINE_8)
    assert cv3.opt.LINE_TYPE == cv2.LINE_8
    
    # Test assertions
    with pytest.raises(AssertionError):
        cv3.opt.draw(thickness=1.5)  # Not an integer
    
    with pytest.raises(AssertionError):
        cv3.opt.draw(pt_radius=-1)  # Negative value
    
    with pytest.raises(AssertionError):
        cv3.opt.draw(scale=0)  # Not positive
    
    with pytest.raises(AssertionError):
        cv3.opt.draw(scale=-1)  # Negative
    
    with pytest.raises(AssertionError):
        cv3.opt.draw(font=999)  # Invalid font
    
    with pytest.raises(AssertionError):
        cv3.opt.draw(line_type=999)  # Invalid line type
    
    # Restore original values
    cv3.opt.THICKNESS = original_thickness
    cv3.opt.COLOR = original_color
    cv3.opt.FONT = original_font
    cv3.opt.PT_RADIUS = original_pt_radius
    cv3.opt.SCALE = original_scale
    cv3.opt.LINE_TYPE = original_line_type


def test_opt_rgb_functions():
    # Test set_rgb function
    original_rgb = cv3.opt.RGB
    cv3.opt.set_bgr()
    assert cv3.opt.RGB == False
    cv3.opt.set_rgb()
    assert cv3.opt.RGB == True
    
    # Restore original value
    cv3.opt.RGB = original_rgb


def test_opt_experimental():
    # Test set_exp function
    original_exp = cv3.opt.EXPERIMENTAL
    cv3.opt.set_exp(True)
    assert cv3.opt.EXPERIMENTAL == True
    cv3.opt.set_exp(False)
    assert cv3.opt.EXPERIMENTAL == False
    
    # Restore original value
    cv3.opt.EXPERIMENTAL = original_exp
