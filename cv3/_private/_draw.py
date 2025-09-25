import cv2
import numpy as np
from typing import List

from .. import opt
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

_THRESHOLD_TYPE_DICT = {
    'binary': cv2.THRESH_BINARY,
    'binary_inv': cv2.THRESH_BINARY_INV,
    'trunc': cv2.THRESH_TRUNC,
    'tozero': cv2.THRESH_TOZERO,
    'tozero_inv': cv2.THRESH_TOZERO_INV
}


def _line_type_flag_match(flag):
    assert flag in _LINE_TYPE_DICT, 'no such flag: "{}". Available: {}'.format(flag, ", ".join(_LINE_TYPE_DICT.keys()))
    return _LINE_TYPE_DICT[flag]

def _font_flag_match(flag):
    assert flag in _FONTS_DICT, 'no such flag: "{}". Available: {}'.format(flag, ", ".join(_FONTS_DICT.keys()))
    return _FONTS_DICT[flag]


def _threshold_type_flag_match(flag):
    assert flag in _THRESHOLD_TYPE_DICT, 'no such flag: "{}". Available: {}'.format(flag, ", ".join(_THRESHOLD_TYPE_DICT.keys()))
    return _THRESHOLD_TYPE_DICT[flag]



def _handle_poly_pts(img, pts, rel=None):
    pts = np.array(pts).reshape(-1)
    pts = _relative_handle(img, *pts, rel=rel)
    pts = np.int32(pts).reshape(-1, 1, 2)
    return pts


def _draw_decorator(func):
    @type_decorator
    def wrapper(img, *args, color=None, copy=False, fill=None, **kwargs):
        if copy:
            img = img.copy()

        color = _process_color(color)

        line_type = kwargs.get('line_type')
        if line_type is None:
            line_type = opt.LINE_TYPE
        elif isinstance(line_type, str):
            line_type = _line_type_flag_match(line_type)
        kwargs['line_type'] = line_type

        t = kwargs.get('t')
        if fill is True:
            if t is not None and t > 0:
                raise ValueError("Cannot specify fill=True and t>0. Use either fill=True for filled shapes or t>0 for outlined shapes.")
            t = cv2.FILLED
        else:
            if fill is False and t is not None and t == -1:
                raise ValueError("Cannot specify fill=False and t=-1. Use either fill=False or t>0 for outlined shapes.")
                
        if t is None:
            t = opt.THICKNESS
        t = round(t)
        kwargs['t'] = t
        
        return func(img, *args, color=color, **kwargs)

    return wrapper


@_draw_decorator
def _rectangle(img, x0, y0, x1, y1, mode='xyxy', rel=None, **kwargs):
    x0, y0, x1, y1 = _handle_rect_coords(img, x0, y0, x1, y1, mode=mode, rel=rel)
    cv2.rectangle(img, (x0, y0), (x1, y1), kwargs['color'], kwargs['t'], lineType=kwargs['line_type'])
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


def _radius_relative_handle(img, r, rel, r_mode):
    """Convert relative radius to absolute radius.
    
    Args:
        img (numpy.ndarray): Input image used to determine dimensions for relative radius.
        r (float): Radius value.
        rel (bool): Relative flag.
        r_mode (str): Mode for relative radius calculation. One of 'w', 'h', 'min', 'max', 'diag'.
        
    Returns:
        int: Absolute radius as integer.
    """
    if not _relative_check(r, rel=rel):
        return round(r)
    
    h, w = img.shape[:2]
    if r_mode == 'w':
        return round(r * w)
    elif r_mode == 'h':
        return round(r * h)
    elif r_mode == 'min':
        return round(r * min(w, h))
    elif r_mode == 'max':
        return round(r * max(w, h))
    elif r_mode == 'diag':
        return round(r * (w**2 + h**2)**0.5)
    else:
        raise ValueError("r_mode must be one of 'w', 'h', 'min', 'max', 'diag'")

@_draw_decorator
def _circle(img, x0, y0, r, rel=None, r_mode='min', **kwargs):
    x0, y0 = _relative_handle(img, x0, y0, rel=rel)
    r = _radius_relative_handle(img, r, rel=rel, r_mode=r_mode)
    cv2.circle(img, (x0, y0), r, kwargs['color'], kwargs['t'], lineType=kwargs['line_type'])
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


@_draw_decorator
def _arrowed_line(img, x0, y0, x1, y1, rel=None, tip_length=None, **kwargs):
    x0, y0, x1, y1 = _relative_handle(img, x0, y0, x1, y1, rel=rel)
    tip_length = tip_length or 0.1
    cv2.arrowedLine(img, (x0, y0), (x1, y1), kwargs['color'], kwargs['t'], kwargs['line_type'], tipLength=tip_length)
    return img


@_draw_decorator
def _ellipse(img, x, y, axes_x, axes_y, angle=0, start_angle=0, end_angle=360, rel=None, **kwargs):
    x, y, axes_x, axes_y = _relative_handle(img, x, y, axes_x, axes_y, rel=rel)
    axes_x, axes_y = round(axes_x), round(axes_y)
    cv2.ellipse(img, (x, y), (axes_x, axes_y), angle, start_angle, end_angle, kwargs['color'], kwargs['t'], lineType=kwargs['line_type'])
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


@_draw_decorator
def _point(img, x0, y0, r=None, rel=None, r_mode='min', **kwargs):
    if r is None:
        r = opt.PT_RADIUS
    kwargs['t'] = -1  # Points are always filled
    return _circle(img, x0, y0, r, rel=rel, r_mode=r_mode, **kwargs)


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
    assert flag in marker_dict or flag in marker_dict.values(), 'no such flag: "{}". Available: {}'.format(flag, ", ".join(marker_dict.keys()))
    return marker_dict.get(flag, flag)


def _get_text_size(text, font=None, scale=None, t=None):
    if font is None:
        font = opt.FONT
    elif isinstance(font, str):
        font = _font_flag_match(font)
    scale = scale or opt.SCALE
    t = t or opt.THICKNESS
    return cv2.getTextSize(str(text), fontFace=font, fontScale=scale, thickness=t)
