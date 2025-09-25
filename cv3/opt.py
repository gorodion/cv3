"""Global configuration options for cv3.

This module provides global configuration options that affect the behavior
of various functions in the cv3 library. These options can be modified
directly or through provided setter functions.

Attributes:
    RGB (bool): Flag indicating whether to use RGB color format (True) or BGR (False).
    FPS (int): Default frames per second for video operations.
    FOURCC (str or int): Default codec for video writing.
    THICKNESS (int): Default line thickness for drawing operations.
    COLOR: Default color for drawing operations.
    FONT: Default font for text drawing operations.
    LINE_TYPE: Default line type for drawing operations.
    THRESHOLD_TYPE: Default threshold type for threshold operations.
    SCALE (float): Default scale factor for drawing operations.
    PT_RADIUS (int): Default point radius for drawing operations.
    EXPERIMENTAL (bool): Flag to enable experimental features.
"""
import cv2
import numpy as np

RGB = True
FPS = 30
FOURCC = 'mp4v'
THICKNESS = 1
COLOR = 255
FONT = cv2.FONT_HERSHEY_SIMPLEX
SCALE = 1
PT_RADIUS = 1
EXPERIMENTAL = False
LINE_TYPE = cv2.LINE_8
THRESHOLD_TYPE = cv2.THRESH_BINARY

def set_rgb():
    """Set the color format to RGB.
    
    This function sets the global RGB flag to True, indicating that
    color values should be interpreted as RGB rather than BGR.
    """
    global RGB
    RGB = True


def set_bgr():
    """Set the color format to BGR.
    
    This function sets the global RGB flag to False, indicating that
    color values should be interpreted as BGR rather than RGB.
    """
    global RGB
    RGB = False


def video(fps=None, fourcc=None):
    """Set default video parameters.
    
    Args:
        fps (int, optional): Default frames per second for video operations.
        fourcc (str or int, optional): Default codec for video writing.
        
    Raises:
        AssertionError: If fps is not positive.
    """
    global FPS, FOURCC
    if fps is not None:
        fps = int(fps)
        assert fps > 0, 'default fps must be greater than 0'
        FPS = fps
    if fourcc is not None:
        if isinstance(fourcc, str):
            assert len(fourcc) == 4, 'if fourcc is str, len(fourcc) must be 4'
            fourcc = cv2.VideoWriter_fourcc(*fourcc)
        FOURCC = fourcc

def draw(thickness=None, color=None, font=None, pt_radius=None, scale=None, line_type=None):
    """Set default drawing parameters.
    
    Args:
        thickness (int, optional): Default line thickness for drawing operations.
        color (optional): Default color for drawing operations.
        font (optional): Default font for text drawing operations.
        pt_radius (int, optional): Default point radius for drawing operations.
        scale (float, optional): Default scale factor for drawing operations.
        line_type (optional): Default line type for drawing operations.
        
    Raises:
        AssertionError: If thickness is not a positive integer.
        AssertionError: If color is not of a supported type.
    """
    global THICKNESS, COLOR, FONT, PT_RADIUS, SCALE, LINE_TYPE
    if thickness is not None:
        assert isinstance(thickness, (int, np.unsignedinteger)), 'default thickness must be positive integer'
        THICKNESS = thickness
    if color is not None:
        assert isinstance(color, (str, int, float, np.unsignedinteger, np.floating, np.ndarray, list, tuple))
        COLOR = color
    if font is not None:
        # Import font values from _private._draw.py
        from ._private._draw import _FONTS_DICT
        assert font in _FONTS_DICT.values(), 'invalid font type'
        FONT = font
    if pt_radius is not None:
        assert isinstance(pt_radius, (int, np.unsignedinteger)), 'default pt_radius must be a non-negative integer'
        assert pt_radius >= 0, 'default pt_radius must be non-negative'
        PT_RADIUS = pt_radius
    if scale is not None:
        assert isinstance(scale, (int, float, np.floating)), 'default scale must be a positive number'
        assert scale > 0, 'default scale must be positive'
        SCALE = scale
    if line_type is not None:
        # Import line type values from _private._draw.py
        from ._private._draw import _LINE_TYPE_DICT
        assert line_type in _LINE_TYPE_DICT.values(), 'invalid line type'
        LINE_TYPE = line_type


def set_exp(exp=True):
    global EXPERIMENTAL
    EXPERIMENTAL = exp
