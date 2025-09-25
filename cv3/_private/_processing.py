import cv2
import numpy as np
from ._utils import type_decorator, _relative_check
from .. import opt
from ._draw import _threshold_type_flag_match


@type_decorator
def _threshold(img: np.ndarray, thr=127, max=None, type=None, rel=None):
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