"""Color space conversion functions.

This module provides functions for converting between different color spaces
commonly used in computer vision and image processing.

Functions:
    cvt_color, cvtColor: Generic color space conversion function.
    
    rgb2bgr, bgr2rgb: Convert between RGB and BGR color spaces.
    
    rgba2bgra, bgra2rgba: Convert between RGBA and BGRA color spaces.
    
    rgb2gray, bgr2gray: Convert color images to grayscale.
    
    gray2rgb, gray2bgr: Convert grayscale images to color.
    
    gray2rgba, gray2bgra: Convert grayscale images to color with alpha channel.
    
    bgr2hsv, rgb2hsv: Convert color images to HSV color space.
    
    hsv2bgr, hsv2rgb: Convert HSV images to color.
    
    rgb, bgr: Aliases for rgb2bgr and bgr2rgb.
    
    rgba, bgra: Aliases for rgba2bgra and bgra2rgba.
"""
from functools import partial
import cv2
import numpy as np
from ._private._color_spaces import _cvt_color

__all__ = [
    'cvt_color', 'cvtColor',
    'rgb2bgr', 'bgr2rgb',
    'rgba2bgra', 'bgra2rgba',
    'rgb2gray', 'bgr2gray',
    'gray2rgb', 'gray2bgr',
    'gray2rgba', 'gray2bgra',
    'bgr2hsv', 'rgb2hsv',
    'hsv2bgr', 'hsv2rgb',
    'rgb', 'bgr', 'rgba', 'bgra'
]


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
    return _cvt_color(img, code)


def rgb2bgr(img):
    """Convert image from RGB to BGR color space.
    
    Args:
        img (numpy.ndarray): Input image in RGB format.
        
    Returns:
        numpy.ndarray: Image in BGR format.
    """
    return cvt_color(img, cv2.COLOR_RGB2BGR)


def bgr2rgb(img):
    """Convert image from BGR to RGB color space.
    
    Args:
        img (numpy.ndarray): Input image in BGR format.
        
    Returns:
        numpy.ndarray: Image in RGB format.
    """
    return cvt_color(img, cv2.COLOR_BGR2RGB)


def rgba2bgra(img):
    """Convert image from RGBA to BGRA color space.
    
    Args:
        img (numpy.ndarray): Input image in RGBA format.
        
    Returns:
        numpy.ndarray: Image in BGRA format.
    """
    return cvt_color(img, cv2.COLOR_RGBA2BGRA)


def bgra2rgba(img):
    """Convert image from BGRA to RGBA color space.
    
    Args:
        img (numpy.ndarray): Input image in BGRA format.
        
    Returns:
        numpy.ndarray: Image in RGBA format.
    """
    return cvt_color(img, cv2.COLOR_BGRA2RGBA)


def rgb2gray(img):
    """Convert image from RGB to grayscale.
    
    Args:
        img (numpy.ndarray): Input image in RGB format.
        
    Returns:
        numpy.ndarray: Grayscale image.
    """
    return cvt_color(img, cv2.COLOR_RGB2GRAY)


def bgr2gray(img):
    """Convert image from BGR to grayscale.
    
    Args:
        img (numpy.ndarray): Input image in BGR format.
        
    Returns:
        numpy.ndarray: Grayscale image.
    """
    return cvt_color(img, cv2.COLOR_BGR2GRAY)


def gray2rgb(img):
    """Convert grayscale image to RGB.
    
    Args:
        img (numpy.ndarray): Input grayscale image.
        
    Returns:
        numpy.ndarray: Image in RGB format.
        
    Raises:
        ValueError: If input image is not grayscale.
    """
    return cvt_color(img, cv2.COLOR_GRAY2RGB)


def gray2bgr(img):
    """Convert grayscale image to BGR.
    
    Args:
        img (numpy.ndarray): Input grayscale image.
        
    Returns:
        numpy.ndarray: Image in BGR format.
        
    Raises:
        ValueError: If input image is not grayscale.
    """
    return cvt_color(img, cv2.COLOR_GRAY2BGR)


def gray2rgba(img):
    """Convert grayscale image to RGBA.
    
    Args:
        img (numpy.ndarray): Input grayscale image.
        
    Returns:
        numpy.ndarray: Image in RGBA format.
        
    Raises:
        ValueError: If input image is not grayscale.
    """
    return cvt_color(img, cv2.COLOR_GRAY2RGBA)


def gray2bgra(img):
    """Convert grayscale image to BGRA.
    
    Args:
        img (numpy.ndarray): Input grayscale image.
        
    Returns:
        numpy.ndarray: Image in BGRA format.
        
    Raises:
        ValueError: If input image is not grayscale.
    """
    return cvt_color(img, cv2.COLOR_GRAY2BGRA)


def bgr2hsv(img):
    """Convert image from BGR to HSV color space.
    
    Args:
        img (numpy.ndarray): Input image in BGR format.
        
    Returns:
        numpy.ndarray: Image in HSV format.
    """
    return cvt_color(img, cv2.COLOR_BGR2HSV)


def rgb2hsv(img):
    """Convert image from RGB to HSV color space.
    
    Args:
        img (numpy.ndarray): Input image in RGB format.
        
    Returns:
        numpy.ndarray: Image in HSV format.
    """
    return cvt_color(img, cv2.COLOR_RGB2HSV)


def hsv2bgr(img):
    """Convert image from HSV to BGR color space.
    
    Args:
        img (numpy.ndarray): Input image in HSV format.
        
    Returns:
        numpy.ndarray: Image in BGR format.
    """
    return cvt_color(img, cv2.COLOR_HSV2BGR)


def hsv2rgb(img):
    """Convert image from HSV to RGB color space.
    
    Args:
        img (numpy.ndarray): Input image in HSV format.
        
    Returns:
        numpy.ndarray: Image in RGB format.
    """
    return cvt_color(img, cv2.COLOR_HSV2RGB)


# Aliases
rgb = rgb2bgr
"""Alias for :func:`rgb2bgr`."""
bgr = bgr2rgb
"""Alias for :func:`bgr2rgb`."""
rgba = rgba2bgra
"""Alias for :func:`rgba2bgra`."""
bgra = bgra2rgba
"""Alias for :func:`bgra2rgba`."""
cvtColor = cvt_color
"""Alias for :func:`cvt_color`."""