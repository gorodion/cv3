"""Image transformation operations.

This module provides functions for transforming images including flipping, rotating,
scaling, shifting, resizing, cropping, and padding.
"""

import cv2

# Import internal functions from _private._transform
from ._private._transform import (
    _vflip, _hflip, _dflip,
    _transform, _rotate, _scale,
    _shift, _xshift, _yshift,
    _resize, _crop, _pad
)

__all__ = [
    'vflip', 'hflip', 'dflip',
    'transform',
    'rotate', 'rotate90', 'rotate180', 'rotate270',
    'scale',
    'shift', 'translate',
    'xshift', 'xtranslate',
    'yshift', 'ytranslate',
    'resize',
    'crop',
    'pad', 'copyMakeBorder',
]

def vflip(img):
    """Flip image vertically (around x-axis).
    
    Args:
        img (numpy.ndarray): Input image.
        
    Returns:
        numpy.ndarray: Vertically flipped image.
        
    Example:
        >>> import cv3
        >>> import numpy as np
        >>> # Create a simple image
        >>> img = np.zeros((100, 100, 3), dtype=np.uint8)
        >>> img[25:75, 25:75] = [255, 255, 255]  # White square
        >>> # Flip vertically
        >>> flipped = cv3.vflip(img)
    """
    return _vflip(img)


def hflip(img):
    """Flip image horizontally (around y-axis).
    
    Args:
        img (numpy.ndarray): Input image.
        
    Returns:
        numpy.ndarray: Horizontally flipped image.
        
    Example:
        >>> import cv3
        >>> import numpy as np
        >>> # Create a simple image
        >>> img = np.zeros((100, 100, 3), dtype=np.uint8)
        >>> img[25:75, 25:75] = [255, 255, 255]  # White square
        >>> # Flip horizontally
        >>> flipped = cv3.hflip(img)
    """
    return _hflip(img)


def dflip(img):
    """Flip image diagonally (around both axes).
    
    Args:
        img (numpy.ndarray): Input image.
        
    Returns:
        numpy.ndarray: Diagonally flipped image.
        
    Example:
        >>> import cv3
        >>> import numpy as np
        >>> # Create a simple image
        >>> img = np.zeros((100, 100, 3), dtype=np.uint8)
        >>> img[25:75, 25:75] = [255, 255, 255]  # White square
        >>> # Flip diagonally
        >>> flipped = cv3.dflip(img)
    """
    return _dflip(img)


def transform(img, angle, scale, inter=cv2.INTER_LINEAR, border=cv2.BORDER_CONSTANT, value=None):
    """Apply affine transformation to image.
    
    Args:
        img (numpy.ndarray): Input image.
        angle (float): Rotation angle in degrees.
        scale (float): Scaling factor.
        inter (int or str, optional): Interpolation method. Can be one of: 'nearest', 'linear', 
            'area', 'cubic', 'lanczos4' or OpenCV flags. Defaults to cv2.INTER_LINEAR.
        border (int or str, optional): Border type. Can be one of: 'constant', 'replicate', 
            'reflect', 'wrap', 'default' or OpenCV flags. Defaults to cv2.BORDER_CONSTANT.
        value: Border color value for constant border type. Defaults to None.
        
    Returns:
        numpy.ndarray: Transformed image.
        
    Example:
        >>> import cv3
        >>> import numpy as np
        >>> # Create a simple image
        >>> img = np.zeros((100, 100, 3), dtype=np.uint8)
        >>> img[25:75, 25:75] = [255, 255, 255]  # White square
        >>> # Rotate 45 degrees and scale by 1.5
        >>> transformed = cv3.transform(img, 45, 1.5)
        >>> # Rotate with custom border color
        >>> transformed = cv3.transform(img, 45, 1.5, value='red')
    """
    return _transform(img, angle, scale, inter=inter, border=border, value=value)


def rotate(img, angle, inter=cv2.INTER_LINEAR, border=cv2.BORDER_CONSTANT, value=None):
    """Rotate image by specified angle.
    
    Args:
        img (numpy.ndarray): Input image.
        angle (float): Rotation angle in degrees.
        inter (int or str, optional): Interpolation method. Can be one of: 'nearest', 'linear', 
            'area', 'cubic', 'lanczos4' or OpenCV flags. Defaults to cv2.INTER_LINEAR.
        border (int or str, optional): Border type. Can be one of: 'constant', 'replicate', 
            'reflect', 'wrap', 'default' or OpenCV flags. Defaults to cv2.BORDER_CONSTANT.
        value: Border color value for constant border type. Defaults to None.
        
    Returns:
        numpy.ndarray: Rotated image.
        
    Example:
        >>> import cv3
        >>> import numpy as np
        >>> # Create a simple image
        >>> img = np.zeros((100, 100, 3), dtype=np.uint8)
        >>> img[25:75, 25:75] = [255, 255, 255]  # White square
        >>> # Rotate 45 degrees
        >>> rotated = cv3.rotate(img, 45)
        >>> # Rotate with custom border color
        >>> rotated = cv3.rotate(img, 45, value=[0, 128, 128])
    """
    return _rotate(img, angle, inter=inter, border=border, value=value)


def scale(img, factor, inter=cv2.INTER_LINEAR, border=cv2.BORDER_CONSTANT, value=None):
    """Scale image by specified factor.
    
    Args:
        img (numpy.ndarray): Input image.
        factor (float): Scaling factor.
        inter (int or str, optional): Interpolation method. Can be one of: 'nearest', 'linear', 
            'area', 'cubic', 'lanczos4' or OpenCV flags. Defaults to cv2.INTER_LINEAR.
        border (int or str, optional): Border type. Can be one of: 'constant', 'replicate', 
            'reflect', 'wrap', 'default' or OpenCV flags. Defaults to cv2.BORDER_CONSTANT.
        value: Border color value for constant border type. Defaults to None.
        
    Returns:
        numpy.ndarray: Scaled image.
        
    Example:
        >>> import cv3
        >>> import numpy as np
        >>> # Create a simple image
        >>> img = np.zeros((100, 100, 3), dtype=np.uint8)
        >>> img[25:75, 25:75] = [255, 255, 255]  # White square
        >>> # Scale by 1.5
        >>> scaled = cv3.scale(img, 1.5)
        >>> # Scale with custom border color
        >>> scaled = cv3.scale(img, 0.5, value='green')
    """
    return _scale(img, factor, inter=inter, border=border, value=value)


def shift(img, x, y, border=cv2.BORDER_CONSTANT, value=None, rel=None):
    """Shift image by x and y pixels.
    
    Args:
        img (numpy.ndarray): Input image.
        x (int or float): Shift in x direction.
        y (int or float): Shift in y direction.
        border (int or str, optional): Border type. Can be one of: 'constant', 'replicate', 
            'reflect', 'wrap', 'default' or OpenCV flags. Defaults to cv2.BORDER_CONSTANT.
        value: Border color value for constant border type. Defaults to None.
        rel (bool, optional): Whether to interpret x and y as relative values. Defaults to None.
        
    Returns:
        numpy.ndarray: Shifted image.
        
    Example:
        >>> import cv3
        >>> import numpy as np
        >>> # Create a simple image
        >>> img = np.zeros((100, 100, 3), dtype=np.uint8)
        >>> img[25:75, 25:75] = [255, 255, 255]  # White square
        >>> # Shift by 10 pixels in x and 20 pixels in y
        >>> shifted = cv3.shift(img, 10, 20)
    """
    return _shift(img, x, y, border=border, value=value, rel=rel)


def xshift(img, x, border=cv2.BORDER_CONSTANT, value=None, rel=None):
    """Shift image horizontally by x pixels.
    
    Args:
        img (numpy.ndarray): Input image.
        x (int or float): Shift in x direction.
        border (int or str, optional): Border type. Can be one of: 'constant', 'replicate', 
            'reflect', 'wrap', 'default' or OpenCV flags. Defaults to cv2.BORDER_CONSTANT.
        value: Border color value for constant border type. Defaults to None.
        rel (bool, optional): Whether to interpret x as relative value. Defaults to None.
        
    Returns:
        numpy.ndarray: Horizontally shifted image.
        
    Example:
        >>> import cv3
        >>> import numpy as np
        >>> # Create a simple image
        >>> img = np.zeros((100, 100, 3), dtype=np.uint8)
        >>> img[25:75, 25:75] = [255, 255, 255]  # White square
        >>> # Shift by 10 pixels in x direction
        >>> shifted = cv3.xshift(img, 10)
    """
    return _xshift(img, x, border=border, value=value, rel=rel)


def yshift(img, y, border=cv2.BORDER_CONSTANT, value=None, rel=None):
    """Shift image vertically by y pixels.
    
    Args:
        img (numpy.ndarray): Input image.
        y (int or float): Shift in y direction.
        border (int or str, optional): Border type. Can be one of: 'constant', 'replicate', 
            'reflect', 'wrap', 'default' or OpenCV flags. Defaults to cv2.BORDER_CONSTANT.
        value: Border color value for constant border type. Defaults to None.
        rel (bool, optional): Whether to interpret y as relative value. Defaults to None.
        
    Returns:
        numpy.ndarray: Vertically shifted image.
        
    Example:
        >>> import cv3
        >>> import numpy as np
        >>> # Create a simple image
        >>> img = np.zeros((100, 100, 3), dtype=np.uint8)
        >>> img[25:75, 25:75] = [255, 255, 255]  # White square
        >>> # Shift by 10 pixels in y direction
        >>> shifted = cv3.yshift(img, 10)
    """
    return _yshift(img, y, border=border, value=value, rel=rel)


def resize(img, width, height, inter=cv2.INTER_LINEAR, rel=None):
    """Resize image to specified dimensions.
    
    Args:
        img (numpy.ndarray): Input image.
        width (int or float): Target width.
        height (int or float): Target height.
        inter (int or str, optional): Interpolation method. Can be one of: 'nearest', 'linear', 
            'area', 'cubic', 'lanczos4' or OpenCV flags. Defaults to cv2.INTER_LINEAR.
        rel (bool, optional): Whether to interpret width and height as relative values. Defaults to None.
        
    Returns:
        numpy.ndarray: Resized image.
        
    Example:
        >>> import cv3
        >>> import numpy as np
        >>> # Create a simple image
        >>> img = np.zeros((100, 100, 3), dtype=np.uint8)
        >>> img[25:75, 25:75] = [255, 255, 255]  # White square
        >>> # Resize to 200x200
        >>> resized = cv3.resize(img, 200, 200)
    """
    return _resize(img, width, height, inter=inter, rel=rel)


def crop(img, x0, y0, x1, y1, mode='xyxy', rel=None, copy=True):
    """Crop image to specified rectangle.
    
    Args:
        img (numpy.ndarray): Input image.
        x0, y0, x1, y1 (int or float): Rectangle coordinates.
        mode (str, optional): Coordinate mode. Can be 'xyxy', 'xywh', 'ccwh'. Defaults to 'xyxy'.
        rel (bool, optional): Whether to interpret coordinates as relative values. Defaults to None.
        copy (bool, optional): Whether to return a copy of the cropped region. Defaults to True.
        
    Returns:
        numpy.ndarray: Cropped image.
        
    Example:
        >>> import cv3
        >>> import numpy as np
        >>> # Create a simple image
        >>> img = np.zeros((100, 100, 3), dtype=np.uint8)
        >>> img[:, :] = [255, 255, 255]  # White image
        >>> # Crop to 50x50 square in the center
        >>> cropped = cv3.crop(img, 25, 25, 75, 75)
        >>> # Crop without copying the data
        >>> cropped_view = cv3.crop(img, 25, 25, 75, 75, copy=False)
    """
    return _crop(img, x0, y0, x1, y1, mode=mode, rel=rel, copy=copy)


def pad(img, y0, y1, x0, x1, border=cv2.BORDER_CONSTANT, value=None, rel=None):
    """Pad image with specified borders.
    
    Args:
        img (numpy.ndarray): Input image.
        y0, y1, x0, x1 (int or float): Padding values for each side.
        border (int or str, optional): Border type. Can be one of: 'constant', 'replicate', 
            'reflect', 'wrap', 'default' or OpenCV flags. Defaults to cv2.BORDER_CONSTANT.
        value: Border color value for constant border type. Defaults to None.
        rel (bool, optional): Whether to interpret padding values as relative. Defaults to None.
        
    Returns:
        numpy.ndarray: Padded image.
        
    Example:
        >>> import cv3
        >>> import numpy as np
        >>> # Create a simple image
        >>> img = np.zeros((100, 100, 3), dtype=np.uint8)
        >>> img[25:75, 25:75] = [255, 255, 255]  # White square
        >>> # Pad with 10 pixels on each side
        >>> padded = cv3.pad(img, 10, 10, 10, 10)
        >>> # Pad with custom border color
        >>> padded = cv3.pad(img, 10, 10, 10, 10, value=[128, 128, 0])
    """
    return _pad(img, y0, y1, x0, x1, border=border, value=value, rel=rel)


def rotate90(img, inter=cv2.INTER_LINEAR, border=cv2.BORDER_CONSTANT, value=None):
    """Rotate image by 90 degrees clockwise.
    
    Args:
        img (numpy.ndarray): Input image.
        inter (int or str, optional): Interpolation method. Can be one of: 'nearest', 'linear',
            'area', 'cubic', 'lanczos4' or OpenCV flags. Defaults to cv2.INTER_LINEAR.
        border (int or str, optional): Border type. Can be one of: 'constant', 'replicate',
            'reflect', 'wrap', 'default' or OpenCV flags. Defaults to cv2.BORDER_CONSTANT.
        value: Border color value for constant border type. Defaults to None.
        
    Returns:
        numpy.ndarray: Rotated image.
        
    Example:
        >>> import cv3
        >>> import numpy as np
        >>> # Create a simple image
        >>> img = np.zeros((100, 100, 3), dtype=np.uint8)
        >>> img[25:75, 25:75] = [255, 255, 255]  # White square
        >>> # Rotate by 90 degrees
        >>> rotated = cv3.rotate90(img)
    """
    return _rotate(img, 90, inter=inter, border=border, value=value)


def rotate180(img, inter=cv2.INTER_LINEAR, border=cv2.BORDER_CONSTANT, value=None):
    """Rotate image by 180 degrees.
    
    Args:
        img (numpy.ndarray): Input image.
        inter (int or str, optional): Interpolation method. Can be one of: 'nearest', 'linear',
            'area', 'cubic', 'lanczos4' or OpenCV flags. Defaults to cv2.INTER_LINEAR.
        border (int or str, optional): Border type. Can be one of: 'constant', 'replicate',
            'reflect', 'wrap', 'default' or OpenCV flags. Defaults to cv2.BORDER_CONSTANT.
        value: Border color value for constant border type. Defaults to None.
        
    Returns:
        numpy.ndarray: Rotated image.
        
    Example:
        >>> import cv3
        >>> import numpy as np
        >>> # Create a simple image
        >>> img = np.zeros((100, 100, 3), dtype=np.uint8)
        >>> img[25:75, 25:75] = [255, 255, 255]  # White square
        >>> # Rotate by 180 degrees
        >>> rotated = cv3.rotate180(img)
    """
    return _rotate(img, 180, inter=inter, border=border, value=value)


def rotate270(img, inter=cv2.INTER_LINEAR, border=cv2.BORDER_CONSTANT, value=None):
    """Rotate image by 270 degrees clockwise (or 90 degrees counter-clockwise).
    
    Args:
        img (numpy.ndarray): Input image.
        inter (int or str, optional): Interpolation method. Can be one of: 'nearest', 'linear',
            'area', 'cubic', 'lanczos4' or OpenCV flags. Defaults to cv2.INTER_LINEAR.
        border (int or str, optional): Border type. Can be one of: 'constant', 'replicate',
            'reflect', 'wrap', 'default' or OpenCV flags. Defaults to cv2.BORDER_CONSTANT.
        value: Border color value for constant border type. Defaults to None.
        
    Returns:
        numpy.ndarray: Rotated image.
        
    Example:
        >>> import cv3
        >>> import numpy as np
        >>> # Create a simple image
        >>> img = np.zeros((100, 100, 3), dtype=np.uint8)
        >>> img[25:75, 25:75] = [255, 255, 255]  # White square
        >>> # Rotate by 270 degrees
        >>> rotated = cv3.rotate270(img)
    """
    return _rotate(img, 270, inter=inter, border=border, value=value)

# Aliases
translate = shift
"""Alias for :func:`shift`."""
xtranslate = xshift
"""Alias for :func:`xshift`."""
ytranslate = yshift
"""Alias for :func:`yshift`."""
copyMakeBorder = pad
"""Alias for :func:`pad`."""

