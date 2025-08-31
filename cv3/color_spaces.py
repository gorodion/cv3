"""Color space conversion functions.

This module provides functions for converting between different color spaces
commonly used in computer vision and image processing.

Functions:
    rgb, bgr: Convert between RGB and BGR color spaces.
    rgba, bgra: Convert between RGBA and BGRA color spaces.
    rgb2gray, bgr2gray: Convert color images to grayscale.
    gray2rgb, gray2bgr: Convert grayscale images to color.
    gray2rgba, gray2bgra: Convert grayscale images to color with alpha channel.
    bgr2hsv, rgb2hsv: Convert color images to HSV color space.
    hsv2bgr, hsv2rgb: Convert HSV images to color.
    cvt_color, cvtColor: Generic color space conversion function.
"""
from functools import partial
import cv2
import numpy as np
from ._utils import type_decorator

__all__ = [
    'rgb', 'bgr', 'rgb2bgr', 'bgr2rgb',
    'rgba', 'bgra', 'rgba2bgra', 'bgra2rgba',
    'rgb2gray',
    'bgr2gray',
    'gray2rgb', 'gray2bgr',
    'gray2rgba', 'gray2bgra',
    'bgr2hsv',
    'rgb2hsv',
    'hsv2bgr',
    'hsv2rgb',
    'cvt_color', 'cvtColor'
]


@type_decorator
def cvt_color(img, code):
    """Convert image between different color spaces.
    
    Args:
        img (numpy.ndarray): Input image.
        code (int): Color space conversion code (e.g., cv2.COLOR_RGB2BGR).
        
    Returns:
        numpy.ndarray: Image in the target color space.
        
    Raises:
        ValueError: If trying to convert a non-grayscale image to RGB/RGBA.
    """
    if code in (cv2.COLOR_GRAY2RGB, cv2.COLOR_GRAY2RGBA):
        if img.ndim == 3 and img.shape[-1] != 1:
            raise ValueError('Image must be grayscale (2 dims)')
    return cv2.cvtColor(img, code=code)


rgb2bgr = bgr2rgb = bgr = rgb = partial(cvt_color, code=cv2.COLOR_RGB2BGR)
rgba2bgra = bgra2rgba = rgba = bgra = partial(cvt_color, code=cv2.COLOR_RGBA2BGRA)
gray2rgb = gray2bgr = partial(cvt_color, code=cv2.COLOR_GRAY2RGB)
gray2rgba = gray2bgra = partial(cvt_color, code=cv2.COLOR_GRAY2RGBA)
bgr2gray = partial(cvt_color, code=cv2.COLOR_BGR2GRAY)
rgb2gray = partial(cvt_color, code=cv2.COLOR_RGB2GRAY)
bgr2hsv = partial(cvt_color, code=cv2.COLOR_BGR2HSV)
rgb2hsv = partial(cvt_color, code=cv2.COLOR_RGB2HSV)
hsv2bgr = partial(cvt_color, code=cv2.COLOR_HSV2BGR)
hsv2rgb = partial(cvt_color, code=cv2.COLOR_HSV2RGB)

cvtColor = cvt_color
"""Alias for cvt_color function."""