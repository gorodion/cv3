"""Internal drawing functions for cv3.

This module contains internal drawing functions used by the public draw.py module.
These functions are not meant to be used directly by users, but rather through
the wrapper functions in draw.py which provide a more user-friendly interface.

Functions:
    _rectangle: Draw a rectangle on an image.
    _polylines: Draw connected line segments on an image.
    _fill_poly: Draw a filled polygon on an image.
    _circle: Draw a circle on an image.
    _line: Draw a line on an image.
    _hline: Draw a horizontal line on an image.
    _vline: Draw a vertical line on an image.
    _text: Draw text on an image.
    _rectangles: Draw multiple rectangles on an image.
    _points: Draw multiple points on an image.
    _arrowed_line: Draw an arrowed line on an image.
    _ellipse: Draw an ellipse on an image.
    _marker: Draw a marker on an image.
    _get_text_size: Calculate the size of a text string.

Constants:
    COLORS: List of named colors available for use in drawing functions.
    _LINE_TYPE_DICT: Dictionary mapping line type names to OpenCV constants.
    _FONTS_DICT: Dictionary mapping font names to OpenCV constants.
"""
import cv2
import numpy as np
from typing import List

from . import opt
from ._utils import (
    type_decorator,
    _relative_check,
    _relative_handle,
    _process_color,
    _handle_rect_coords,
    COLORS_RGB_DICT
)

COLORS = list(COLORS_RGB_DICT)


_LINE_TYPE_DICT = {
    'filled': cv2.FILLED,
    'line_4': cv2.LINE_4,
    'line_8': cv2.LINE_8,
    'line_aa': cv2.LINE_AA
}


_FONTS_DICT = {
    'simplex': cv2.FONT_HERSHEY_SIMPLEX,
    'plain': cv2.FONT_HERSHEY_PLAIN,
    'duplex': cv2.FONT_HERSHEY_DUPLEX,
    'complex': cv2.FONT_HERSHEY_COMPLEX,
    'triplex': cv2.FONT_HERSHEY_TRIPLEX,
    'complex_small': cv2.FONT_HERSHEY_COMPLEX_SMALL,
    'script_simplex': cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
    'script_complex': cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
    'italic': cv2.FONT_ITALIC
}


def _line_type_flag_match(flag):
    assert flag in _LINE_TYPE_DICT, f'no such flag: "{flag}". Available: {", ".join(_LINE_TYPE_DICT.keys())}'
    return _LINE_TYPE_DICT[flag]


def _font_flag_match(flag):
    assert flag in _FONTS_DICT, f'no such flag: "{flag}". Available: {", ".join(_FONTS_DICT.keys())}'
    return _FONTS_DICT[flag]


def _handle_poly_pts(img, pts, rel=None):
    pts = np.array(pts).reshape(-1)
    pts = _relative_handle(img, *pts, rel=rel)
    pts = np.int32(pts).reshape(-1, 1, 2)
    return pts


def _draw_decorator(func):
    @type_decorator
    def wrapper(img, *args, color=None, line_type=cv2.LINE_8, copy=False, **kwargs):
        if copy:
            img = img.copy()

        color = _process_color(color)

        if isinstance(line_type, str):
            line_type = _line_type_flag_match(line_type)

        kwargs['t'] = round(kwargs.get('t', opt.THICKNESS))

        return func(img, *args, color=color, line_type=line_type, **kwargs)

    return wrapper


@_draw_decorator
def _rectangle(img, x0, y0, x1, y1, mode='xyxy', rel=None, fill=None, **kwargs):
    x0, y0, x1, y1 = _handle_rect_coords(img, x0, y0, x1, y1, mode=mode, rel=rel)
    # Handle fill parameter and validation
    if fill is True:
        thickness = cv2.FILLED
    elif fill is False and kwargs['t'] == -1:
        raise ValueError("Cannot specify fill=False and t=-1. Use either fill=False or t>0 for outlined rectangles.")
    elif fill is None:
        thickness = kwargs['t']
    else:
        thickness = kwargs['t']
    cv2.rectangle(img, (x0, y0), (x1, y1), kwargs['color'], thickness, lineType=kwargs['line_type'])
    return img


@_draw_decorator
def _polylines(img, pts, is_closed=False, rel=None, **kwargs):
    pts = _handle_poly_pts(img, pts, rel=rel)
    cv2.polylines(img, [pts], is_closed, kwargs['color'], kwargs['t'], lineType=kwargs['line_type'])
    return img


@_draw_decorator
def _fill_poly(img, pts, rel=None, **kwargs):
    pts = _handle_poly_pts(img, pts, rel=rel)
    cv2.fillPoly(img, [pts], kwargs['color'])
    return img


@_draw_decorator
def _circle(img, x0, y0, r, rel=None, fill=None, **kwargs):
    x0, y0 = _relative_handle(img, x0, y0, rel=rel)
    r = round(r)
    # Handle fill parameter and validation
    if fill is True:
        thickness = cv2.FILLED
    elif fill is False and kwargs['t'] == -1:
        raise ValueError("Cannot specify fill=False and t=-1. Use either fill=False or t>0 for outlined circles.")
    elif fill is None:
        thickness = kwargs['t']
    else:
        thickness = kwargs['t']
    cv2.circle(img, (x0, y0), r, kwargs['color'], thickness, lineType=kwargs['line_type'])
    return img


@_draw_decorator
def _line(img, x0, y0, x1, y1, rel=None, **kwargs):
    x0, y0, x1, y1 = _relative_handle(img, x0, y0, x1, y1, rel=rel)
    cv2.line(img, (x0, y0), (x1, y1), kwargs['color'], kwargs['t'], lineType=kwargs['line_type'])
    return img


@_draw_decorator
def _hline(img, y, rel=None, **kwargs):
    h, w = img.shape[:2]
    y = round(y * h if _relative_check(y, rel=rel) else y)
    cv2.line(img, (0, y), (w, y), kwargs['color'], kwargs['t'], lineType=kwargs['line_type'])
    return img


@_draw_decorator
def _vline(img, x, rel=None, **kwargs):
    h, w = img.shape[:2]
    x = round(x * w if _relative_check(x, rel=rel) else x)
    cv2.line(img, (x, 0), (x, h), kwargs['color'], kwargs['t'], lineType=kwargs['line_type'])
    return img


@_draw_decorator
def _text(img, text, x=0.5, y=0.5, font=None, scale=None, flip=False, rel=None, **kwargs):
    if font is None:
        font = opt.FONT
    elif isinstance(font, str):
        font = _font_flag_match(font)
    scale = scale or opt.SCALE
    x, y = _relative_handle(img, x, y, rel=rel)
    cv2.putText(
        img,
        str(text),
        (x, y),
        fontFace=font,
        fontScale=scale,
        color=kwargs['color'],
        thickness=kwargs['t'],
        lineType=kwargs['line_type'],
        bottomLeftOrigin=flip
    )
    return img


@type_decorator
def _rectangles(img, rects: List[List], color=None, t=None, line_type=None, fill=None, copy=False) -> np.array:
    kwargs = {}
    if color is not None:
        kwargs['color'] = color
    if t is not None:
        kwargs['t'] = t
    if line_type is not None:
        kwargs['line_type'] = line_type
    if fill is not None:
        kwargs['fill'] = fill
    if copy:
        kwargs['copy'] = copy
    for rect in rects:
        img = _rectangle(img, *rect, **kwargs)
    return img


@type_decorator
def _points(img: np.array, pts: List[List], color=None, r=None, copy=False) -> np.array:
    kwargs = {}
    if color is not None:
        kwargs['color'] = color
    if copy:
        kwargs['copy'] = copy
    # Handle the radius parameter specially since _circle expects it as a positional arg
    if r is None:
        r = opt.PT_RADIUS
    for pt in pts:
        img = _circle(img, pt[0], pt[1], r, t=-1, **kwargs)
    return img


@_draw_decorator
def _arrowed_line(img, x0, y0, x1, y1, rel=None, tip_length=None, **kwargs):
    x0, y0, x1, y1 = _relative_handle(img, x0, y0, x1, y1, rel=rel)
    tip_length = tip_length or 0.1
    cv2.arrowedLine(img, (x0, y0), (x1, y1), kwargs['color'], kwargs['t'], kwargs['line_type'], tipLength=tip_length)
    return img


@_draw_decorator
def _ellipse(img, x, y, axes_x, axes_y, angle=0, start_angle=0, end_angle=360, rel=None, fill=None, **kwargs):
    x, y, axes_x, axes_y = _relative_handle(img, x, y, axes_x, axes_y, rel=rel)
    axes_x, axes_y = round(axes_x), round(axes_y)
    # Handle fill parameter and validation
    if fill is True:
        thickness = cv2.FILLED
    elif fill is False and kwargs['t'] == -1:
        raise ValueError("Cannot specify fill=False and t=-1. Use either fill=False or t>0 for outlined ellipses.")
    elif fill is None:
        thickness = kwargs['t']
    else:
        thickness = kwargs['t']
    cv2.ellipse(img, (x, y), (axes_x, axes_y), angle, start_angle, end_angle, kwargs['color'], thickness, lineType=kwargs['line_type'])
    return img


@_draw_decorator
def _marker(img, x, y, marker_type=None, marker_size=None, rel=None, **kwargs):
    x, y = _relative_handle(img, x, y, rel=rel)
    marker_type = marker_type or cv2.MARKER_CROSS
    marker_size = marker_size or 20
    if isinstance(marker_type, str):
        marker_type = _marker_flag_match(marker_type)
    cv2.drawMarker(img, (x, y), kwargs['color'], markerType=marker_type, markerSize=marker_size, thickness=kwargs['t'], line_type=kwargs['line_type'])
    return img


def _marker_flag_match(flag):
    marker_dict = {
        'cross': cv2.MARKER_CROSS,
        'tilted_cross': cv2.MARKER_TILTED_CROSS,
        'star': cv2.MARKER_STAR,
        'diamond': cv2.MARKER_DIAMOND,
        'square': cv2.MARKER_SQUARE,
        'triangle_up': cv2.MARKER_TRIANGLE_UP,
        'triangle_down': cv2.MARKER_TRIANGLE_DOWN
    }
    assert flag in marker_dict or flag in marker_dict.values(), f'no such flag: "{flag}". Available: {", ".join(marker_dict.keys())}'
    return marker_dict.get(flag, flag)


def _get_text_size(text, font=None, scale=None, t=None):
    if font is None:
        font = opt.FONT
    elif isinstance(font, str):
        font = _font_flag_match(font)
    scale = scale or opt.SCALE
    t = t or opt.THICKNESS
    return cv2.getTextSize(str(text), fontFace=font, fontScale=scale, thickness=t)