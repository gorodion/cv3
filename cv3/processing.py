"""Image processing operations.

This module provides functions for basic image processing operations
such as thresholding.

Functions:
    threshold: Apply binary threshold to grayscale images.
"""
import cv2
import numpy as np

from ._utils import type_decorator
__all__ = [
    'threshold'
]


# TODO flags
@type_decorator
def threshold(img: np.ndarray, thr=127, max=255):
    """Apply binary threshold to a grayscale image.
    
    Args:
        img (numpy.ndarray): Input grayscale image.
        thr (int, optional): Threshold value. Defaults to 127.
        max (int, optional): Maximum value to use with the THRESH_BINARY thresholding.
            Defaults to 255.
            
    Returns:
        numpy.ndarray: Thresholded image.
        
    Raises:
        AssertionError: If the input image is not a grayscale image.
        
    Note:
        This function applies binary thresholding using cv2.THRESH_BINARY.
        Pixels with values greater than the threshold are set to the maximum value,
        and pixels with values less than or equal to the threshold are set to 0.
        
    Example:
        >>> import cv3
        >>> import numpy as np
        >>> # Create a simple grayscale image
        >>> img = np.zeros((100, 100), dtype=np.uint8)
        >>> img[25:75, 25:75] = 128  # Gray square
        >>> # Apply threshold
        >>> thresh = cv3.threshold(img, 100, 255)
    """
    assert img.ndim == 2, '`img` must be gray image'
    # TODO if img.max() < 1
    _, thresh = cv2.threshold(img, thr, max, cv2.THRESH_BINARY)
    return thresh
