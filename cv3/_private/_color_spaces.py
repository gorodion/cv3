import cv2
import numpy as np
from ._utils import type_decorator


@type_decorator
def _cvt_color(img, code):
    if code in (cv2.COLOR_GRAY2RGB, cv2.COLOR_GRAY2RGBA):
        if img.ndim == 3 and img.shape[-1] != 1:
            raise ValueError('Image must be grayscale (2 dims)')
    return cv2.cvtColor(img, code=code)