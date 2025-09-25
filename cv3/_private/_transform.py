"""Internal transform functions for cv3.

This module contains the internal implementations of transform functions,
wrapped with type_decorator. These functions are not meant to be used directly
by users, but are used by the public API in transform.py.
"""

import cv2
import numpy as np
import warnings
from functools import partial

from ._utils import type_decorator, _relative_check, _relative_handle, _process_color, _handle_rect_coords
from ..utils import xywh2xyxy, ccwh2xyxy, yyxx2xyxy

# Interpolation methods dictionary
_INTER_DICT = {
    'nearest': cv2.INTER_NEAREST,
    'linear': cv2.INTER_LINEAR,
    'area': cv2.INTER_AREA,
    'cubic': cv2.INTER_CUBIC,
    'lanczos4': cv2.INTER_LANCZOS4
}

# Border types dictionary
_BORDER_DICT = {
    'constant': cv2.BORDER_CONSTANT,
    'replicate': cv2.BORDER_REPLICATE,
    'reflect': cv2.BORDER_REFLECT,
    'wrap': cv2.BORDER_WRAP,
    'default': cv2.BORDER_DEFAULT,
    # 'transparent': cv2.BORDER_TRANSPARENT
}


def _inter_flag_match(flag):
    """Convert string interpolation flag to OpenCV flag constant.
    
    Args:
        flag (str): String flag name. Can be one of: 'nearest', 'linear', 'area', 'cubic', 'lanczos4'.
        
    Returns:
        int: OpenCV interpolation flag constant.
        
    Raises:
        AssertionError: If flag is not one of the valid options.
    """
    assert flag in _INTER_DICT, 'no such flag: "{}". Available: {}'.format(flag, ", ".join(_INTER_DICT.keys()))
    return _INTER_DICT[flag]


def _border_flag_match(flag):
    """Convert string border flag to OpenCV flag constant.
    
    Args:
        flag (str): String flag name. Can be one of: 'constant', 'replicate', 'reflect', 'wrap', 'default'.
        
    Returns:
        int: OpenCV border flag constant.
        
    Raises:
        AssertionError: If flag is not one of the valid options.
    """
    assert flag in _BORDER_DICT, 'no such flag: "{}". Available: {}'.format(flag, ", ".join(_BORDER_DICT.keys()))
    return _BORDER_DICT[flag]


def _border_value_check(border, value):
    """Check and process border and value parameters.
    
    Args:
        border (str or int): Border type.
        value: Border value for constant border type.
        
    Returns:
        tuple: Processed border and value.
    """
    if isinstance(border, str):
        border = _border_flag_match(border)
    if value is not None:
        value = _process_color(value)
        if border != cv2.BORDER_CONSTANT:
            warnings.warn('`value` parameter is not used when `border` is not cv2.BORDER_CONSTANT')
    return border, value


@type_decorator
def _vflip(img):
    """Flip image vertically (around x-axis).
    
    Args:
        img (numpy.ndarray): Input image.
        
    Returns:
        numpy.ndarray: Vertically flipped image.
    """
    return cv2.flip(img, 0)


@type_decorator
def _hflip(img):
    """Flip image horizontally (around y-axis).
    
    Args:
        img (numpy.ndarray): Input image.
        
    Returns:
        numpy.ndarray: Horizontally flipped image.
    """
    return cv2.flip(img, 1)


@type_decorator
def _dflip(img):
    """Flip image diagonally (around both axes).
    
    Args:
        img (numpy.ndarray): Input image.
        
    Returns:
        numpy.ndarray: Diagonally flipped image.
    """
    return cv2.flip(img, -1)


@type_decorator
def _transform(img, angle, scale, inter=cv2.INTER_LINEAR, border=cv2.BORDER_CONSTANT, value=None):
    """Apply affine transformation to image.
    
    Args:
        img (numpy.ndarray): Input image.
        angle (float): Rotation angle in degrees.
        scale (float): Scaling factor.
        inter (int or str, optional): Interpolation method. Defaults to cv2.INTER_LINEAR.
        border (int or str, optional): Border type. Defaults to cv2.BORDER_CONSTANT.
        value: Border value for constant border type. Defaults to None.
        
    Returns:
        numpy.ndarray: Transformed image.
    """
    if isinstance(inter, str):
        inter = _inter_flag_match(inter)
    border, value = _border_value_check(border, value)
    rot_mat = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), angle, scale)
    result = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=inter, borderMode=border, borderValue=value)
    return result


def _rotate(img, angle, inter=cv2.INTER_LINEAR, border=cv2.BORDER_CONSTANT, value=None):
    """Rotate image by specified angle.
    
    Args:
        img (numpy.ndarray): Input image.
        angle (float): Rotation angle in degrees.
        inter (int or str, optional): Interpolation method. Defaults to cv2.INTER_LINEAR.
        border (int or str, optional): Border type. Defaults to cv2.BORDER_CONSTANT.
        value: Border value for constant border type. Defaults to None.
        
    Returns:
        numpy.ndarray: Rotated image.
    """
    return _transform(img, angle, 1, inter=inter, border=border, value=value)


def _scale(img, factor, inter=cv2.INTER_LINEAR, border=cv2.BORDER_CONSTANT, value=None):
    """Scale image by specified factor.
    
    Args:
        img (numpy.ndarray): Input image.
        factor (float): Scaling factor.
        inter (int or str, optional): Interpolation method. Defaults to cv2.INTER_LINEAR.
        border (int or str, optional): Border type. Defaults to cv2.BORDER_CONSTANT.
        value: Border value for constant border type. Defaults to None.
        
    Returns:
        numpy.ndarray: Scaled image.
    """
    return _transform(img, 0, factor, inter=inter, border=border, value=value)


@type_decorator
def _shift(img, x, y, border=cv2.BORDER_CONSTANT, value=None, rel=None):
    """Shift image by x and y pixels.
    
    Args:
        img (numpy.ndarray): Input image.
        x (int or float): Shift in x direction.
        y (int or float): Shift in y direction.
        border (int or str, optional): Border type. Defaults to cv2.BORDER_CONSTANT.
        value: Border value for constant border type. Defaults to None.
        rel (bool, optional): Whether to interpret x and y as relative values. Defaults to None.
        
    Returns:
        numpy.ndarray: Shifted image.
    """
    x, y = _relative_handle(img, x, y, rel=rel)
    transMat = np.float32([[1, 0, x], [0, 1, y]])
    dimensions = (img.shape[1], img.shape[0])
    border, value = _border_value_check(border, value)
    return cv2.warpAffine(img, transMat, dimensions, borderMode=border, borderValue=value)


def _xshift(img, x, border=cv2.BORDER_CONSTANT, value=None, rel=None):
    """Shift image horizontally by x pixels.
    
    Args:
        img (numpy.ndarray): Input image.
        x (int or float): Shift in x direction.
        border (int or str, optional): Border type. Defaults to cv2.BORDER_CONSTANT.
        value: Border value for constant border type. Defaults to None.
        rel (bool, optional): Whether to interpret x as relative value. Defaults to None.
        
    Returns:
        numpy.ndarray: Horizontally shifted image.
    """
    h, w = img.shape[:2]
    x = round(x * w if _relative_check(x, rel=rel) else x)
    return _shift(img, x, 0, border=border, value=value, rel=False)


def _yshift(img, y, border=cv2.BORDER_CONSTANT, value=None, rel=None):
    """Shift image vertically by y pixels.
    
    Args:
        img (numpy.ndarray): Input image.
        y (int or float): Shift in y direction.
        border (int or str, optional): Border type. Defaults to cv2.BORDER_CONSTANT.
        value: Border value for constant border type. Defaults to None.
        rel (bool, optional): Whether to interpret y as relative value. Defaults to None.
        
    Returns:
        numpy.ndarray: Vertically shifted image.
    """
    h, w = img.shape[:2]
    y = round(y * h if _relative_check(y, rel=rel) else y)
    return _shift(img, 0, y, border=border, value=value, rel=False)


@type_decorator
def _resize(img, width, height, inter=cv2.INTER_LINEAR, rel=None):
    """Resize image to specified dimensions.
    
    Args:
        img (numpy.ndarray): Input image.
        width (int or float): Target width.
        height (int or float): Target height.
        inter (int or str, optional): Interpolation method. Defaults to cv2.INTER_LINEAR.
        rel (bool, optional): Whether to interpret width and height as relative values. Defaults to None.
        
    Returns:
        numpy.ndarray: Resized image.
        
    Raises:
        ValueError: If width or height is zero.
    """
    if isinstance(inter, str):
        inter = _inter_flag_match(inter)
    width, height = _relative_handle(img, width, height, rel=rel)
    if width == 0 or height == 0:
        if not rel:
            warnings.warn('Try to set `rel` to True')
        raise ValueError('Width or height have zero size')
    return cv2.resize(img, (width, height), interpolation=inter)


@type_decorator
def _crop(img, x0, y0, x1, y1, mode='xyxy', rel=None, copy=True):
    """Crop image to specified rectangle.
    
    Args:
        img (numpy.ndarray): Input image.
        x0, y0, x1, y1 (int or float): Rectangle coordinates.
        mode (str, optional): Coordinate mode. Defaults to 'xyxy'.
        rel (bool, optional): Whether to interpret coordinates as relative values. Defaults to None.
        copy (bool, optional): Whether to return a copy of the cropped region. Defaults to True.
        
    Returns:
        numpy.ndarray: Cropped image.
    """
    x0, y0, x1, y1 = _handle_rect_coords(img, x0, y0, x1, y1, mode=mode, rel=rel)

    x0, x1 = min(x0, x1), max(x0, x1)
    y0, y1 = min(y0, y1), max(y0, y1)

    x0 = max(x0, 0)
    y0 = max(y0, 0)

    if y1 == y0 or x1 == x0:
        if not rel:
            warnings.warn('zero-size array. Try to set `rel` to True')
    return img[y0:y1, x0:x1].copy() if copy else img[y0:y1, x0:x1]


@type_decorator
def _pad(img, y0, y1, x0, x1, border=cv2.BORDER_CONSTANT, value=None, rel=None):
    """Pad image with specified borders.
    
    Args:
        img (numpy.ndarray): Input image.
        y0, y1, x0, x1 (int or float): Padding values for each side.
        border (int or str, optional): Border type. Defaults to cv2.BORDER_CONSTANT.
        value: Border value for constant border type. Defaults to None.
        rel (bool, optional): Whether to interpret padding values as relative. Defaults to None.
        
    Returns:
        numpy.ndarray: Padded image.
    """
    border, value = _border_value_check(border, value)
    x0, y0, x1, y1 = _relative_handle(img, x0, y0, x1, y1, rel=rel)
    return cv2.copyMakeBorder(img, y0, y1, x0, x1, borderType=border, value=value)

