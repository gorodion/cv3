"""Drawing functions for computer vision applications.

This module provides a collection of drawing functions that wrap OpenCV's drawing
capabilities with a more user-friendly interface. All functions support relative
coordinates, automatic color handling, and other conveniences.

Functions:
    rectangle: Draw a rectangle on an image.
    polylines: Draw connected line segments on an image.
    fill_poly: Draw a filled polygon on an image.
    circle: Draw a circle on an image.
    point: Draw a point (filled circle) on an image.
    points: Draw multiple points on an image.
    line: Draw a line on an image.
    hline: Draw a horizontal line on an image.
    vline: Draw a vertical line on an image.
    text: Draw text on an image.
    rectangles: Draw multiple rectangles on an image.
    arrow: Draw an arrowed line on an image (experimental).
    ellipse: Draw an ellipse on an image (experimental).
    marker: Draw a marker on an image (experimental).
    getTextSize: Calculate the size of a text string (experimental).

Constants:
    COLORS: List of named colors available for use in drawing functions.
"""
import warnings
import numpy as np
from typing import List

from . import opt
from ._private._draw import (
    _rectangle,
    _polylines,
    _fill_poly,
    _circle,
    _point,
    _line,
    _hline,
    _vline,
    _text,
    _arrowed_line,
    _ellipse,
    _marker,
    _get_text_size,
    COLORS
)

__all__ = [
    'rectangle',
    'polylines',
    'fill_poly',
    'circle',
    'point',
    'points',
    'line', 'hline', 'vline',
    'text', 'putText',
    'rectangles', 'rect', 'rects',
    'arrow',
    'ellipse',
    'marker',
    'getTextSize',
    'poly', 'polygon',
    'COLORS'
]

def rectangle(img, x0, y0, x1, y1, mode='xyxy', rel=None, color=None, t=None, line_type=None, fill=None, copy=False):
    """Draw a rectangle on an image.

    Args:
        img (numpy.ndarray): Input image to draw on.
        x0 (int or float): X-coordinate of the first point.
        y0 (int or float): Y-coordinate of the first point.
        x1 (int or float): X-coordinate of the second point.
        y1 (int or float): Y-coordinate of the second point.
        mode (str, optional): Coordinate mode. One of 'xyxy', 'xywh', 'ccwh'. Defaults to 'xyxy'.
        rel (bool, optional): Whether to use relative coordinates. Defaults to None.
        color: Color of the rectangle (default: opt.COLOR).
        t: Thickness of the rectangle lines (default: opt.THICKNESS).
        line_type: Type of line for drawing (default: opt.LINE_TYPE).
        fill (bool, optional): Whether to fill the rectangle. If True, draws a filled rectangle
            regardless of thickness. If False, draws an outlined rectangle. If None, uses
            the thickness parameter to determine fill behavior. Defaults to None.
        copy (bool): Whether to copy the image before drawing (default: False).

    Returns:
        numpy.ndarray: Image with the rectangle drawn on it.

    Note:
        The coordinate modes are:
        - 'xyxy': Two corner points (x0, y0) and (x1, y1)
        - 'xywh': Top-left corner (x0, y0) and width (x1), height (y1)
        - 'ccwh': Center point (x0, y0) and width (x1), height (y1)

        Relative coordinates are in the range [0, 1] where 0 is the top/left
        and 1 is the bottom/right of the image.

        When fill=True, the rectangle is filled regardless of the thickness value.
        When fill=False, the rectangle is outlined with the specified thickness.
        When fill=None (default), the rectangle is filled if t=-1, otherwise outlined.

    Example:
        >>> import cv3
        >>> img = cv3.zeros(100, 100, 3)
        >>> # Draw a rectangle using absolute coordinates
        >>> img = cv3.rectangle(img, 10, 10, 90, 90, color='red', t=2)
        >>> # Draw a filled rectangle
        >>> img = cv3.rectangle(img, 20, 20, 80, 80, color='blue', fill=True)
        >>> # Draw a rectangle using relative coordinates
        >>> img = cv3.rectangle(img, 0.2, 0.2, 0.8, 0.8, rel=True, color='green')
        >>> # Draw a rectangle using xywh mode
        >>> img = cv3.rectangle(img, 10, 10, 80, 80, mode='xywh', color='yellow')
    """
    return _rectangle(img, x0, y0, x1, y1, mode=mode, rel=rel, color=color, copy=copy, t=t, line_type=line_type, fill=fill)


def polylines(img, pts, is_closed=False, rel=None, color=None, t=None, line_type=None, copy=False):
    """Draw polylines on an image.

    Args:
        img (numpy.ndarray): Input image to draw on.
        pts (array-like): Points defining the polylines. Can be a numpy array,
            list of lists, or tuple of tuples with shape (N, 2) where N is
            the number of points.
        is_closed (bool, optional): Whether to close the polyline by connecting
            the last point to the first. Defaults to False.
        rel (bool, optional): Whether to use relative coordinates. Defaults to None.
        color: Color of the polylines (default: opt.COLOR).
        t: Thickness of the lines (default: opt.THICKNESS).
        line_type: Type of line for drawing (default: opt.LINE_TYPE).
        copy (bool): Whether to copy the image before drawing (default: False).

    Returns:
        numpy.ndarray: Image with the polylines drawn on it.

    Note:
        Points can be specified in various formats:
        - List of [x, y] coordinates: [[x1, y1], [x2, y2], ...]
        - Tuple of (x, y) coordinates: ((x1, y1), (x2, y2), ...)
        - Flattened list: [x1, y1, x2, y2, ...]
        - Numpy array with shape (N, 2) or (N, 1, 2)

        Relative coordinates are in the range [0, 1] where 0 is the top/left
        and 1 is the bottom/right of the image.

    Example:
        >>> import cv3
        >>> img = cv3.zeros(100, 100, 3)
        >>> points = [[10, 10], [50, 20], [90, 10], [90, 90]]
        >>> # Draw open polylines
        >>> img = cv3.polylines(img, points, color='red', t=2)
        >>> # Draw closed polylines
        >>> img = cv3.polylines(img, points, is_closed=True, color='blue')
    """
    return _polylines(img, pts, is_closed=is_closed, rel=rel, color=color, copy=copy, t=t, line_type=line_type)


def fill_poly(img, pts, rel=None, color=None, copy=False):
    """Draw a filled polygon on an image.

    Args:
        img (numpy.ndarray): Input image to draw on.
        pts (array-like): Points defining the polygon. Can be a numpy array,
            list of lists, or tuple of tuples with shape (N, 2) where N is
            the number of points.
        rel (bool, optional): Whether to use relative coordinates. Defaults to None.
        color: Color to fill the polygon (default: opt.COLOR).
        copy (bool): Whether to copy the image before drawing (default: False).

    Returns:
        numpy.ndarray: Image with the filled polygon drawn on it.

    Note:
        Points can be specified in various formats:
        - List of [x, y] coordinates: [[x1, y1], [x2, y2], ...]
        - Tuple of (x, y) coordinates: ((x1, y1), (x2, y2), ...)
        - Flattened list: [x1, y1, x2, y2, ...]
        - Numpy array with shape (N, 2) or (N, 1, 2)

        Relative coordinates are in the range [0, 1] where 0 is the top/left
        and 1 is the bottom/right of the image.

    Example:
        >>> import cv3
        >>> img = cv3.zeros(100, 100, 3)
        >>> points = [[10, 10], [50, 20], [90, 10], [90, 90]]
        >>> # Draw a filled polygon
        >>> img = cv3.fill_poly(img, points, color='red')
    """
    return _fill_poly(img, pts, rel=rel, color=color, copy=copy)


def circle(img, x0, y0, r, rel=None, r_mode='min', color=None, t=None, line_type=None, fill=None, copy=False):
    """Draw a circle on an image.

    Args:
        img (numpy.ndarray): Input image to draw on.
        x0 (int or float): X-coordinate of the circle center.
        y0 (int or float): Y-coordinate of the circle center.
        r (int or float): Radius of the circle.
        rel (bool, optional): Whether to use relative coordinates. Defaults to None.
        r_mode (str, optional): Mode for relative radius calculation. One of 'w', 'h', 'min', 'max', 'diag'.
            Only used when rel=True for the radius. Defaults to 'min'.
            - 'w': Relative to image width
            - 'h': Relative to image height
            - 'min': Relative to minimum of width and height
            - 'max': Relative to maximum of width and height
            - 'diag': Relative to image diagonal
        color: Color of the circle (default: opt.COLOR).
        t: Thickness of the circle line. Use -1 or cv2.FILLED for filled circle (default: opt.THICKNESS).
        line_type: Type of line for drawing (default: opt.LINE_TYPE).
        fill (bool, optional): Whether to fill the circle. If True, draws a filled circle
            regardless of thickness. If False, draws an outlined circle. If None, uses
            the thickness parameter to determine fill behavior. Defaults to None.
        copy (bool): Whether to copy the image before drawing (default: False).

    Returns:
        numpy.ndarray: Image with the circle drawn on it.

    Note:
        Relative coordinates are in the range [0, 1] where 0 is the top/left
        and 1 is the bottom/right of the image.

        To draw a filled circle, you can either:
        - Use t=-1 or set the thickness parameter to -1
        - Use fill=True

        When fill=True, the circle is filled regardless of the thickness value.
        When fill=False, the circle is outlined with the specified thickness.
        When fill=None (default), the circle is filled if t=-1, otherwise outlined.

    Example:
        >>> import cv3
        >>> img = cv3.zeros(100, 100, 3)
        >>> # Draw a circle with outline
        >>> img = cv3.circle(img, 50, 50, 30, color='red', t=2)
        >>> # Draw a filled circle using fill parameter
        >>> img = cv3.circle(img, 80, 80, 15, color='blue', fill=True)
        >>> # Draw a filled circle using thickness parameter
        >>> img = cv3.circle(img, 20, 20, 10, color='green', t=-1)
        >>> # Draw a circle with relative radius based on image width
        >>> img = cv3.circle(img, 0.5, 0.5, 0.2, rel=True, r_mode='w', color='yellow')
    """
    return _circle(img, x0, y0, r, rel=rel, r_mode=r_mode, color=color, copy=copy, t=t, line_type=line_type, fill=fill)


def point(img, x0, y0, r=None, rel=None, r_mode='min', color=None, copy=False):
    """Draw a point (filled circle) on an image.

    Args:
        img (numpy.ndarray): Input image to draw on.
        x0 (int or float): X-coordinate of the point center.
        y0 (int or float): Y-coordinate of the point center.
        r (int or float, optional): Radius of the point. Defaults to opt.PT_RADIUS.
        rel (bool, optional): Whether to use relative coordinates. Defaults to None.
        r_mode (str, optional): Mode for relative radius calculation. One of 'w', 'h', 'min', 'max', 'diag'.
            Only used when rel=True for the radius. Defaults to 'min'.
            - 'w': Relative to image width
            - 'h': Relative to image height
            - 'min': Relative to minimum of width and height
            - 'max': Relative to maximum of width and height
            - 'diag': Relative to image diagonal
        color: Color of the point (default: opt.COLOR).
        copy (bool): Whether to copy the image before drawing (default: False).

    Returns:
        numpy.ndarray: Image with the point drawn on it.

    Note:
        This function draws a filled circle (point) on the image. The thickness
        parameter 't' is not used for points as they are always filled.

        Relative coordinates are in the range [0, 1] where 0 is the top/left
        and 1 is the bottom/right of the image.

    Example:
        >>> import cv3
        >>> img = cv3.zeros(100, 100, 3)
        >>> # Draw a point
        >>> img = cv3.point(img, 50, 50, color='red')
        >>> # Draw a point with custom radius
        >>> img = cv3.point(img, 80, 80, r=5, color='blue')
        >>> # Draw a point with relative radius based on image width
        >>> img = cv3.point(img, 0.5, 0.5, r=0.1, rel=True, r_mode='w', color='green')
    """
    return _point(img, x0, y0, r=r, rel=rel, r_mode=r_mode, color=color, copy=copy)


def line(img, x0, y0, x1, y1, rel=None, color=None, t=None, line_type=None, copy=False):
    """Draw a line on an image.

    Args:
        img (numpy.ndarray): Input image to draw on.
        x0 (int or float): X-coordinate of the first point.
        y0 (int or float): Y-coordinate of the first point.
        x1 (int or float): X-coordinate of the second point.
        y1 (int or float): Y-coordinate of the second point.
        rel (bool, optional): Whether to use relative coordinates. Defaults to None.
        color: Color of the line (default: opt.COLOR).
        t: Thickness of the line (default: opt.THICKNESS).
        line_type: Type of line for drawing (default: opt.LINE_TYPE).
        copy (bool): Whether to copy the image before drawing (default: False).

    Returns:
        numpy.ndarray: Image with the line drawn on it.

    Note:
        Relative coordinates are in the range [0, 1] where 0 is the top/left
        and 1 is the bottom/right of the image.

    Example:
        >>> import cv3
        >>> img = cv3.zeros(100, 100, 3)
        >>> # Draw a line
        >>> img = cv3.line(img, 10, 10, 90, 90, color='red', t=2)
        >>> # Draw a line using relative coordinates
        >>> img = cv3.line(img, 0.2, 0.2, 0.8, 0.8, rel=True, color='blue')
    """
    return _line(img, x0, y0, x1, y1, rel=rel, color=color, copy=copy, t=t, line_type=line_type)


def hline(img, y, rel=None, color=None, t=None, line_type=None, copy=False):
    """Draw a horizontal line on an image.

    Args:
        img (numpy.ndarray): Input image to draw on.
        y (int or float): Y-coordinate of the horizontal line.
        rel (bool, optional): Whether to use relative coordinates. Defaults to None.
        color: Color of the line (default: opt.COLOR).
        t: Thickness of the line (default: opt.THICKNESS).
        line_type: Type of line for drawing (default: opt.LINE_TYPE).
        copy (bool): Whether to copy the image before drawing (default: False).

    Returns:
        numpy.ndarray: Image with the horizontal line drawn on it.

    Note:
        Relative coordinates are in the range [0, 1] where 0 is the top
        and 1 is the bottom of the image.

    Example:
        >>> import cv3
        >>> img = cv3.zeros(100, 100, 3)
        >>> # Draw a horizontal line in the middle
        >>> img = cv3.hline(img, 50, color='red', t=2)
        >>> # Draw a horizontal line using relative coordinates
        >>> img = cv3.hline(img, 0.75, rel=True, color='blue')
    """
    return _hline(img, y, rel=rel, color=color, copy=copy, t=t, line_type=line_type)


def vline(img, x, rel=None, color=None, t=None, line_type=None, copy=False):
    """Draw a vertical line on an image.

    Args:
        img (numpy.ndarray): Input image to draw on.
        x (int or float): X-coordinate of the vertical line.
        rel (bool, optional): Whether to use relative coordinates. Defaults to None.
        color: Color of the line (default: opt.COLOR).
        t: Thickness of the line (default: opt.THICKNESS).
        line_type: Type of line for drawing (default: opt.LINE_TYPE).
        copy (bool): Whether to copy the image before drawing (default: False).

    Returns:
        numpy.ndarray: Image with the vertical line drawn on it.

    Note:
        Relative coordinates are in the range [0, 1] where 0 is the left
        and 1 is the right of the image.

    Example:
        >>> import cv3
        >>> img = cv3.zeros(100, 100, 3)
        >>> # Draw a vertical line in the middle
        >>> img = cv3.vline(img, 50, color='red', t=2)
        >>> # Draw a vertical line using relative coordinates
        >>> img = cv3.vline(img, 0.75, rel=True, color='blue')
    """
    return _vline(img, x, rel=rel, color=color, copy=copy, t=t, line_type=line_type)


def text(img, text_str, x=0.5, y=0.5, font=None, scale=None, flip=False, rel=None, color=None, t=None, line_type=None, copy=False):
    """Draw text on an image.

    Args:
        img (numpy.ndarray): Input image to draw on.
        text_str (str): Text string to be drawn.
        x (int or float, optional): X-coordinate of the bottom-left corner of the text.
            Defaults to 0.5 (center).
        y (int or float, optional): Y-coordinate of the bottom-left corner of the text.
            Defaults to 0.5 (center).
        font (int or str, optional): Font type. Can be an OpenCV font flag or string.
            Available string options: 'simplex', 'plain', 'duplex', 'complex',
            'triplex', 'complex_small', 'script_simplex', 'script_complex', 'italic'.
            Defaults to opt.FONT.
        scale (float, optional): Font scale factor that is multiplied by the
            font-specific base size. Defaults to opt.SCALE.
        flip (bool, optional): If True, the text is rendered upside down.
            Defaults to False.
        rel (bool, optional): Whether to use relative coordinates. Defaults to None.
        color: Color of the text (default: opt.COLOR).
        t: Thickness of the text strokes (default: opt.THICKNESS).
        line_type: Type of line for drawing (default: opt.LINE_TYPE).
        copy (bool): Whether to copy the image before drawing (default: False).

    Returns:
        numpy.ndarray: Image with the text drawn on it.

    Note:
        Relative coordinates are in the range [0, 1] where (0, 0) is the top-left
        and (1, 1) is the bottom-right of the image.

        The default position (x=0.5, y=0.5) places the text near the center of the image.

    Example:
        >>> import cv3
        >>> img = cv3.zeros(100, 100, 3)
        >>> # Draw text at the center
        >>> img = cv3.text(img, 'Hello', color='red')
        >>> # Draw text with custom position and font
        >>> img = cv3.text(img, 'World', x=10, y=50, font='complex', scale=1.2, color='blue')
        >>> # Draw flipped text
        >>> img = cv3.text(img, 'Flipped', x=50, y=80, flip=True, color='green')
    """
    return _text(img, text_str, x=x, y=y, rel=rel, color=color, copy=copy, t=t, line_type=line_type, scale=scale, font=font, flip=flip)


def rectangles(img: np.array, rects: List[List], *args, **kwargs) -> np.array:
    """Draw multiple rectangles on an image. See :func:`rectangle` for more details.

    Args:
        img (numpy.ndarray): Input image to draw on.
        rects (List[List]): List of rectangles, where each rectangle is a list
            of parameters to pass to the rectangle function.
        *args: Additional arguments to pass to the :func:`rectangle` function.
        **kwargs: Additional keyword arguments to pass to the :func:`rectangle` function.

    Returns:
        numpy.ndarray: Image with all rectangles drawn on it.

    Note:
        Each rectangle in the rects list should contain the parameters needed
        for the rectangle function (x0, y0, x1, y1, etc.).

        The coordinate modes are:
        - 'xyxy': Two corner points (x0, y0) and (x1, y1)
        - 'xywh': Top-left corner (x0, y0) and width (x1), height (y1)
        - 'ccwh': Center point (x0, y0) and width (x1), height (y1)

        Relative coordinates are in the range [0, 1] where 0 is the top/left
        and 1 is the bottom/right of the image.

        When fill=True, the rectangles are filled regardless of the thickness value.
        When fill=False, the rectangles are outlined with the specified thickness.
        When fill=None (default), the rectangles are filled if t=-1, otherwise outlined.

    Example:
        >>> import cv3
        >>> img = cv3.zeros(100, 100, 3)
        >>> # Draw multiple rectangles
        >>> rectangles = [
        ...     [10, 10, 30, 30],
        ...     [40, 40, 60, 60],
        ...     [70, 70, 90, 90]
        ... ]
        >>> img = cv3.rectangles(img, rectangles, color='red', t=2)
        >>> # Draw multiple filled rectangles
        >>> filled_rectangles = [
        ...     [15, 15, 35, 35],
        ...     [45, 45, 65, 65]
        ... ]
        >>> img = cv3.rectangles(img, filled_rectangles, color='blue', fill=True)
    """
    for rect in rects:
        img = rectangle(img, *rect, *args, **kwargs)
    return img


def points(img: np.array, pts: List[List], *args, **kwargs) -> np.array:
    """Draw multiple points on an image. See :func:`point` for more details.

    Args:
        img (numpy.ndarray): Input image to draw on.
        pts (List[List]): List of points, where each point is a list
            of parameters to pass to the point function.
        *args: Additional arguments to pass to the :func:`point` function.
        **kwargs: Additional keyword arguments to pass to the :func:`point` function.

    Returns:
        numpy.ndarray: Image with all points drawn on it.

    Note:
        Each point in the pts list should contain the parameters needed
        for the point function (x0, y0, r, etc.).

        Relative coordinates are in the range [0, 1] where 0 is the top/left
        and 1 is the bottom/right of the image.

    Example:
        >>> import cv3
        >>> img = cv3.zeros(100, 100, 3)
        >>> # Draw multiple points
        >>> points = [
        ...     [10, 10],
        ...     [40, 40],
        ...     [70, 70]
        ... ]
        >>> img = cv3.points(img, points, color='red')
        >>> # Draw multiple points with relative radius based on image width
        >>> points = [
        ...     [0.2, 0.2],
        ...     [0.5, 0.5],
        ...     [0.8, 0.8]
        ... ]
        >>> img = cv3.points(img, points, r=0.1, rel=True, r_mode='w', color='blue')
    """
    for pt in pts:
        img = _point(img, *pt, *args, **kwargs)
    return img


def arrow(img, x0, y0, x1, y1, rel=None, color=None, t=None, line_type=None, tip_length=None, copy=False):
    """Draw an arrowed line on an image.
    
    This is an experimental function. To use it, set experimental mode with cv3.opt.set_exp().

    Args:
        img (numpy.ndarray): Input image to draw on.
        x0 (int or float): X-coordinate of the starting point.
        y0 (int or float): Y-coordinate of the starting point.
        x1 (int or float): X-coordinate of the ending point.
        y1 (int or float): Y-coordinate of the ending point.
        rel (bool, optional): Whether to use relative coordinates. Defaults to None.
        color: Color of the arrow (default: opt.COLOR).
        t: Thickness of the arrow (default: opt.THICKNESS).
        line_type: Type of line for drawing (default: opt.LINE_TYPE).
        tip_length (float, optional): The length of the arrow tip in relation to the arrow length.
            Defaults to 0.1.
        copy (bool): Whether to copy the image before drawing (default: False).

    Returns:
        numpy.ndarray: Image with the arrow drawn on it.

    Note:
        Relative coordinates are in the range [0, 1] where 0 is the top/left
        and 1 is the bottom/right of the image.

    Example:
        >>> import cv3
        >>> img = cv3.zeros(100, 100, 3)
        >>> # Draw an arrow
        >>> img = cv3.arrow(img, 10, 10, 90, 90, color='red', t=2)
        >>> # Draw an arrow with custom tip length
        >>> img = cv3.arrow(img, 20, 20, 80, 80, color='blue', tip_length=0.2)
    """
    if not opt.EXPERIMENTAL:
        raise RuntimeError("This function is experimental. Use opt.set_exp() to enable it.")
    
    return _arrowed_line(img, x0, y0, x1, y1, rel=rel, color=color, copy=copy, t=t, line_type=line_type, tip_length=tip_length)


def ellipse(img, x, y, axes_x, axes_y, angle=0, start_angle=0, end_angle=360, rel=None, color=None, t=None,
            line_type=None, fill=None, copy=False):
    """Draw an ellipse on an image.
    
    This is an experimental function. To use it, set experimental mode with cv3.opt.set_exp().

    Args:
        img (numpy.ndarray): Input image to draw on.
        x (int or float): X-coordinate of the ellipse center.
        y (int or float): Y-coordinate of the ellipse center.
        axes_x (int or float): Half of the size of the ellipse main axis x.
        axes_y (int or float): Half of the size of the ellipse main axis y.
        angle (float, optional): Ellipse rotation angle in degrees. Defaults to 0.
        start_angle (float, optional): Starting angle of the elliptic arc in degrees. Defaults to 0.
        end_angle (float, optional): Ending angle of the elliptic arc in degrees. Defaults to 360.
        rel (bool, optional): Whether to use relative coordinates. Defaults to None.
        color: Color of the ellipse (default: opt.COLOR).
        t: Thickness of the ellipse line. Use -1 or cv2.FILLED for filled ellipse (default: opt.THICKNESS).
        line_type: Type of line for drawing (default: opt.LINE_TYPE).
        fill (bool, optional): Whether to fill the ellipse. If True, draws a filled ellipse
            regardless of thickness. If False, draws an outlined ellipse. If None, uses
            the thickness parameter to determine fill behavior. Defaults to None.
        copy (bool): Whether to copy the image before drawing (default: False).

    Returns:
        numpy.ndarray: Image with the ellipse drawn on it.

    Note:
        Relative coordinates are in the range [0, 1] where 0 is the top/left
        and 1 is the bottom/right of the image.

        To draw a filled ellipse, you can either:
        - Use t=-1 or set the thickness parameter to -1
        - Use fill=True

        When fill=True, the ellipse is filled regardless of the thickness value.
        When fill=False, the ellipse is outlined with the specified thickness.
        When fill=None (default), the ellipse is filled if t=-1, otherwise outlined.

    Example:
        >>> import cv3
        >>> img = cv3.zeros(100, 100, 3)
        >>> # Draw an ellipse with outline
        >>> img = cv3.ellipse(img, 50, 50, 30, 20, color='red', t=2)
        >>> # Draw a filled ellipse using fill parameter
        >>> img = cv3.ellipse(img, 80, 80, 15, 10, color='blue', fill=True)
        >>> # Draw a filled ellipse using thickness parameter
        >>> img = cv3.ellipse(img, 20, 20, 10, 5, color='green', t=-1)
    """
    if not opt.EXPERIMENTAL:
        raise RuntimeError("This function is experimental. Use opt.set_exp() to enable it.")
    
    return _ellipse(img, x, y, axes_x, axes_y, angle=angle, start_angle=start_angle, end_angle=end_angle,
                    rel=rel, color=color, copy=copy, t=t, line_type=line_type, fill=fill)


def marker(img, x, y, marker_type=None, marker_size=None, rel=None, color=None, t=None, line_type=None, copy=False):
    """Draw a marker on an image.
    
    This is an experimental function. To use it, set experimental mode with cv3.opt.set_exp().

    Args:
        img (numpy.ndarray): Input image to draw on.
        x (int or float): X-coordinate of the marker position.
        y (int or float): Y-coordinate of the marker position.
        marker_type (int or str, optional): The specific type of marker you want to use.
            Can be one of: 'cross', 'tilted_cross', 'star', 'diamond', 'square', 'triangle_up', 'triangle_down'.
            Defaults to cv2.MARKER_CROSS.
        marker_size (int, optional): The length of the marker axis. Defaults to 20.
        rel (bool, optional): Whether to use relative coordinates. Defaults to None.
        color: Color of the marker (default: opt.COLOR).
        t: Thickness of the marker lines (default: opt.THICKNESS).
        line_type: Type of line for drawing (default: opt.LINE_TYPE).
        copy (bool): Whether to copy the image before drawing (default: False).

    Returns:
        numpy.ndarray: Image with the marker drawn on it.

    Note:
        Relative coordinates are in the range [0, 1] where 0 is the top/left
        and 1 is the bottom/right of the image.

    Example:
        >>> import cv3
        >>> img = cv3.zeros(100, 100, 3)
        >>> # Draw a cross marker
        >>> img = cv3.marker(img, 50, 50, color='red', t=2)
        >>> # Draw a star marker with custom size
        >>> img = cv3.marker(img, 80, 80, marker_type='star', marker_size=30, color='blue')
    """
    if not opt.EXPERIMENTAL:
        raise RuntimeError("This function is experimental. Use opt.set_exp() to enable it.")
    
    return _marker(img, x, y, marker_type=marker_type, marker_size=marker_size, rel=rel, color=color, copy=copy, t=t, line_type=line_type)


def getTextSize(text, font=None, scale=None, t=None):
    """Calculate the width and height of a text string.
    
    This is an experimental function. To use it, set experimental mode with cv3.opt.set_exp().

    Args:
        text (str): Input text string.
        font (int or str, optional): Font type. Can be an OpenCV font flag or string.
            Available string options: 'simplex', 'plain', 'duplex', 'complex',
            'triplex', 'complex_small', 'script_simplex', 'script_complex', 'italic'.
            Defaults to opt.FONT.
        scale (float, optional): Font scale factor that is multiplied by the
            font-specific base size. Defaults to opt.SCALE.
        t (int, optional): Thickness of the lines used to draw a text. Defaults to opt.THICKNESS.

    Returns:
        tuple: A tuple containing:
            - Size: The size of a box that contains the specified text.
            - baseline (int): y-coordinate of the baseline relative to the bottom-most text point.

    Example:
        >>> import cv3
        >>> # Get text size
        >>> text_size, baseline = cv3.getTextSize('Hello World', font='simplex', scale=1.2, t=2)
        >>> print(f"Text size: {text_size}, Baseline: {baseline}")
    """
    if not opt.EXPERIMENTAL:
        raise RuntimeError("This function is experimental. Use opt.set_exp() to enable it.")
    
    return _get_text_size(text, font=font, scale=scale, t=t)


# Aliases
putText = text
"""Alias for :func:`text`."""
rect = rectangle
"""Alias for :func:`rectangles`."""
rects = rectangles
"""Alias for :func:`rectangles`."""
poly = polylines
"""Alias for :func:`polylines`."""
polygon = polylines
"""Alias for :func:`polylines`."""
