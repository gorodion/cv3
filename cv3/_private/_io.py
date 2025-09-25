import cv2
import warnings


def _imread_flag_match(flag):
    assert flag in ('color', 'gray', 'alpha', 'unchanged')
    if flag == 'color':
        flag = cv2.IMREAD_COLOR
    elif flag == 'gray':
        flag = cv2.IMREAD_GRAYSCALE
    elif flag == 'unchanged':
        flag = cv2.IMREAD_UNCHANGED
    elif flag == 'alpha':
        warnings.warn('Flag name "alpha" deprecated. Please use "unchanged"')
        flag = cv2.IMREAD_UNCHANGED
    return flag


def _is_ascii(s):
    return all(ord(c) < 128 for c in s)