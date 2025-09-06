"""Image processing operations.

This module provides functions for basic image processing operations
such as thresholding.

Functions:
    threshold: Apply binary threshold to grayscale images.
"""
import cv2
import numpy as np
from ._utils import type_decorator, _relative_check, _relative_handle
from . import opt
from ._draw import _threshold_type_flag_match

__all__ = [
    'threshold'
]



@type_decorator
def threshold(img: np.ndarray, thr=127, max=None, type=None, rel=None):
    """Apply threshold to a grayscale image.
    
    Args:
        img (numpy.ndarray): Input grayscale image.
        thr (int or float, optional): Threshold value. Defaults to 127.
        max (int or float, optional): Maximum value to use with the thresholding.
            If None, defaults to 255. Defaults to None.
        type (int or str, optional): Threshold type. Can be an OpenCV threshold flag or string.
            Available string options: 'binary', 'binary_inv', 'trunc', 'tozero', 'tozero_inv'.
            If None, defaults to opt.THRESHOLD_TYPE. Defaults to None.
        rel (bool, optional): Whether to use relative threshold value. Defaults to None.
            
    Returns:
        numpy.ndarray: Thresholded image.
        
    Raises:
        AssertionError: If the input image is not a grayscale image.
        
    Note:
        This function applies thresholding using the specified threshold type.
        Pixels are processed according to the chosen thresholding method.
        
        Relative threshold values are in the range [0, 1] where 0 is the minimum
        and 1 is the maximum pixel value in the image.
        
    Example:
        >>> import cv3
        >>> import numpy as np
        >>> # Create a simple grayscale image
        >>> img = np.zeros((100, 100), dtype=np.uint8)
        >>> img[25:75, 25:75] = 128  # Gray square
        >>> # Apply threshold with default type
        >>> thresh = cv3.threshold(img, 100)
        >>> # Apply threshold with custom type
        >>> thresh = cv3.threshold(img, 100, type='binary_inv')
        >>> # Apply threshold with relative threshold value
        >>> thresh = cv3.threshold(img, 0.5, rel=True)
    """
    assert img.ndim == 2, '`img` must be gray image'
    
    # Handle max value
    if max is None:
        max = 255
    # Handle threshold type
    if type is None:
        type = opt.THRESHOLD_TYPE
    elif isinstance(type, str):
        type = _threshold_type_flag_match(type)
    else:
        # Validate flag type
        from ._draw import _THRESHOLD_TYPE_DICT
        assert type in _THRESHOLD_TYPE_DICT.values(), 'invalid threshold type flag: {}'.format(type)
    
    # Handle relative threshold value
    if _relative_check(thr, rel=rel):
        thr = thr * img.max()
    
    _, thresh = cv2.threshold(img, thr, max, type)
    return thresh

