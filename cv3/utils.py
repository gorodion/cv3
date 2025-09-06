"""Coordinate conversion utilities.

This module provides functions for converting between different coordinate
representations commonly used in computer vision and image processing.

Functions:
    xywh2xyxy: Convert from (x, y, width, height) to (x0, y0, x1, y1) format.
    xyxy2xywh: Convert from (x0, y0, x1, y1) to (x, y, width, height) format.
    ccwh2xyxy: Convert from (center_x, center_y, width, height) to (x0, y0, x1, y1) format.
    xyxy2ccwh: Convert from (x0, y0, x1, y1) to (center_x, center_y, width, height) format.
    yyxx2xyxy: Convert from (y0, y1, x0, x1) to (x0, y0, x1, y1) format.
    rel2abs: Convert relative coordinates to absolute coordinates.
    abs2rel: Convert absolute coordinates to relative coordinates.
"""


def xywh2xyxy(x0, y0, w, h):
    """Convert from (x, y, width, height) to (x0, y0, x1, y1) format.
    
    Args:
        x0 (float): X-coordinate of the top-left corner.
        y0 (float): Y-coordinate of the top-left corner.
        w (float): Width of the rectangle.
        h (float): Height of the rectangle.
        
    Returns:
        tuple: (x0, y0, x1, y1) coordinates of the rectangle.
    """
    x1 = x0 + w
    y1 = y0 + h
    return x0, y0, x1, y1


def xyxy2xywh(x0, y0, x1, y1):
    """Convert from (x0, y0, x1, y1) to (x, y, width, height) format.
    
    Args:
        x0 (float): X-coordinate of the top-left corner.
        y0 (float): Y-coordinate of the top-left corner.
        x1 (float): X-coordinate of the bottom-right corner.
        y1 (float): Y-coordinate of the bottom-right corner.
        
    Returns:
        tuple: (x, y, width, height) representation of the rectangle.
    """
    w = x1 - x0
    h = y1 - y0
    return x0, y0, w, h


def ccwh2xyxy(xc, yc, w, h):
    """Convert from (center_x, center_y, width, height) to (x0, y0, x1, y1) format.
    
    Args:
        xc (float): X-coordinate of the center.
        yc (float): Y-coordinate of the center.
        w (float): Width of the rectangle.
        h (float): Height of the rectangle.
        
    Returns:
        tuple: (x0, y0, x1, y1) coordinates of the rectangle.
    """
    x0 = xc - w / 2
    x1 = xc + w / 2
    y0 = yc - h / 2
    y1 = yc + h / 2
    return x0, y0, x1, y1


def xyxy2ccwh(x0, y0, x1, y1):
    """Convert from (x0, y0, x1, y1) to (center_x, center_y, width, height) format.
    
    Args:
        x0 (float): X-coordinate of the top-left corner.
        y0 (float): Y-coordinate of the top-left corner.
        x1 (float): X-coordinate of the bottom-right corner.
        y1 (float): Y-coordinate of the bottom-right corner.
        
    Returns:
        tuple: (center_x, center_y, width, height) representation of the rectangle.
    """
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2
    w = abs(x1 - x0)
    h = abs(y1 - y0)
    return cx, cy, w, h


def yyxx2xyxy(y0, y1, x0, x1):
    """Convert from (y0, y1, x0, x1) to (x0, y0, x1, y1) format.
    
    Args:
        y0 (float): First y-coordinate.
        y1 (float): Second y-coordinate.
        x0 (float): First x-coordinate.
        x1 (float): Second x-coordinate.
        
    Returns:
        tuple: (x0, y0, x1, y1) coordinates.
    """
    # ¯\_(ツ)_/¯
    return x0, y0, x1, y1

def rel2abs(*coords, width, height):
    """Convert relative coordinates to absolute coordinates.
    
    Args:
        *coords: Iterable of coordinates in the form (x0, y0, x1, y1, ..., xn, yn).
        width (int): Width of the image or reference frame.
        height (int): Height of the image or reference frame.
        
    Yields:
        int: Absolute coordinates rounded to integers.
        
    Raises:
        AssertionError: If the number of coordinates is not even.
    """
    assert len(coords) % 2 == 0
    for x, y in zip(*[iter(coords)] * 2):
        yield round(x * width)
        yield round(y * height)


def abs2rel(*coords, width, height):
    """Convert absolute coordinates to relative coordinates.
    
    Args:
        *coords: Iterable of coordinates in the form (x0, y0, x1, y1, ..., xn, yn).
        width (int): Width of the image or reference frame.
        height (int): Height of the image or reference frame.
        
    Yields:
        float: Relative coordinates in the range [0, 1].
        
    Raises:
        AssertionError: If the number of coordinates is not even.
    """
    assert len(coords) % 2 == 0
    for x, y in zip(*[iter(coords)] * 2):
        yield x / width
        yield y / height
