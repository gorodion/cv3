"""Image creation functions.

This module provides functions for creating images with various initial values
such as zeros, ones, random values, etc. All functions return numpy arrays
with uint8 data type suitable for image processing.

Functions:
    zeros, zeros_like: Create arrays filled with zeros.
    ones, ones_like: Create arrays filled with ones.
    full, full_like: Create arrays filled with a specified value.
    empty, empty_like: Create uninitialized arrays.
    white, white_like: Create arrays filled with white pixels (255).
    random, rand, randn, randint: Create arrays filled with random values.
"""
import numpy as np

__all__ = [
    'zeros',
    'zeros_like',
    'ones',
    'ones_like',
    'full',
    'full_like',
    'empty',
    'empty_like',
    'white',
    'white_like',
    'black',
    'black_like',
    'random',
    'rand',
    'randn',
    'randint'
]


def zeros(*args):
    """Create an array filled with zeros.
    
    Args:
        *args: Dimensions of the array. Can be passed as separate arguments
               (e.g., zeros(100, 100, 3)) or as a tuple (e.g., zeros((100, 100, 3))).
    
    Returns:
        numpy.ndarray: Array filled with zeros of type uint8.
    
    Example:
        >>> import cv3
        >>> img = cv3.zeros(100, 100, 3)  # Create a 100x100x3 image filled with zeros
        >>> print(img.shape)
        (100, 100, 3)
        >>> print(img.dtype)
        uint8
    """
    return np.zeros(args, np.uint8)


def zeros_like(img):
    """Create an array of zeros with the same shape and type as the input array.
    
    Args:
        img (numpy.ndarray): Input array whose shape and type will be used
                             to create the new array.
    
    Returns:
        numpy.ndarray: Array filled with zeros having the same shape and type
                       as the input array.
    
    Example:
        >>> import cv3
        >>> original = cv3.ones(50, 50, 3)
        >>> new_img = cv3.zeros_like(original)  # Create array with same shape as original
        >>> print(new_img.shape)
        (50, 50, 3)
    """
    return np.zeros_like(img, np.uint8)


def ones(*args):
    """Create an array filled with ones.
    
    Args:
        *args: Dimensions of the array. Can be passed as separate arguments
               (e.g., ones(100, 100, 3)) or as a tuple (e.g., ones((100, 100, 3))).
    
    Returns:
        numpy.ndarray: Array filled with ones of type uint8.
    
    Example:
        >>> import cv3
        >>> img = cv3.ones(100, 100, 3)  # Create a 100x100x3 image filled with ones
        >>> print(img.shape)
        (100, 100, 3)
    """
    return np.ones(args, np.uint8)


def ones_like(img):
    """Create an array of ones with the same shape and type as the input array.
    
    Args:
        img (numpy.ndarray): Input array whose shape and type will be used
                             to create the new array.
    
    Returns:
        numpy.ndarray: Array filled with ones having the same shape and type
                       as the input array.
    
    Example:
        >>> import cv3
        >>> original = cv3.zeros(50, 50, 3)
        >>> new_img = cv3.ones_like(original)  # Create array with same shape as original
        >>> print(new_img.shape)
        (50, 50, 3)
    """
    return np.ones_like(img, np.uint8)


def full(*args, value):
    """Create an array filled with a specified value.
    
    Args:
        *args: Dimensions of the array. Can be passed as separate arguments
               (e.g., full(100, 100, 3, value=128)) or as a tuple.
        value: The value to fill the array with.
    
    Returns:
        numpy.ndarray: Array filled with the specified value of type uint8.
    
    Example:
        >>> import cv3
        >>> img = cv3.full(100, 100, 3, value=128)  # Create image filled with 128
        >>> print(img[0, 0])  # All pixels have value 128
        [128 128 128]
    """
    return np.full(args, value, np.uint8)


def full_like(img, value):
    """Create an array filled with a specified value, with the same shape and type as the input array.
    
    Args:
        img (numpy.ndarray): Input array whose shape and type will be used
                             to create the new array.
        value: The value to fill the array with.
    
    Returns:
        numpy.ndarray: Array filled with the specified value having the same shape and type
                       as the input array.
    
    Example:
        >>> import cv3
        >>> original = cv3.zeros(50, 50, 3)
        >>> new_img = cv3.full_like(original, value=255)  # Create array with same shape, filled with 255
        >>> print(new_img.shape)
        (50, 50, 3)
    """
    return np.full_like(img, value, np.uint8)


def empty(*args):
    """Create an uninitialized array.
    
    Args:
        *args: Dimensions of the array. Can be passed as separate arguments
               (e.g., empty(100, 100, 3)) or as a tuple.
    
    Returns:
        numpy.ndarray: Uninitialized array of type uint8.
    
    Note:
        The values in the returned array are uninitialized and may contain
        arbitrary data. For performance reasons, the array is not initialized
        to zero.
    
    Example:
        >>> import cv3
        >>> img = cv3.empty(100, 100, 3)  # Create uninitialized 100x100x3 image
        >>> print(img.shape)
        (100, 100, 3)
    """
    return np.empty(args, np.uint8)


def empty_like(img):
    """Create an uninitialized array with the same shape and type as the input array.
    
    Args:
        img (numpy.ndarray): Input array whose shape and type will be used
                             to create the new array.
    
    Returns:
        numpy.ndarray: Uninitialized array having the same shape and type
                       as the input array.
    
    Note:
        The values in the returned array are uninitialized and may contain
        arbitrary data. For performance reasons, the array is not initialized
        to zero.
    
    Example:
        >>> import cv3
        >>> original = cv3.zeros(50, 50, 3)
        >>> new_img = cv3.empty_like(original)  # Create uninitialized array with same shape
        >>> print(new_img.shape)
        (50, 50, 3)
    """
    return np.empty_like(img, np.uint8)


def white(*args):
    """Create an array filled with white pixels (255).
    
    Args:
        *args: Dimensions of the array. Can be passed as separate arguments
               (e.g., white(100, 100, 3)) or as a tuple.
    
    Returns:
        numpy.ndarray: Array filled with 255 (white) of type uint8.
    
    Example:
        >>> import cv3
        >>> img = cv3.white(100, 100, 3)  # Create a white 100x100x3 image
        >>> print(img[0, 0])  # All pixels are white
        [255 255 255]
    """
    return full(*args, value=255)


def white_like(img):
    """Create an array filled with white pixels (255), with the same shape and type as the input array.
    
    Args:
        img (numpy.ndarray): Input array whose shape and type will be used
                             to create the new array.
    
    Returns:
        numpy.ndarray: Array filled with 255 (white) having the same shape and type
                       as the input array.
    
    Example:
        >>> import cv3
        >>> original = cv3.zeros(50, 50, 3)
        >>> white_img = cv3.white_like(original)  # Create white image with same shape
        >>> print(white_img.shape)
        (50, 50, 3)
    """
    return full_like(img, value=255)


def random(*args):
    """Create an array filled with random values between 0 and 255.
    
    Args:
        *args: Dimensions of the array. Can be passed as separate arguments
               (e.g., random(100, 100, 3)) or as a tuple.
    
    Returns:
        numpy.ndarray: Array filled with random integers [0, 255] of type uint8.
    
    Example:
        >>> import cv3
        >>> img = cv3.random(100, 100, 3)  # Create 100x100x3 image with random values
        >>> print(img.shape)
        (100, 100, 3)
        >>> print(0 <= img.min() <= img.max() <= 255)  # All values in valid range
        True
    """
    return np.random.randint(0, 256, args, np.uint8)

# Aliases
black = zeros
"""Alias for :func:`zeros`."""
black_like = zeros_like
"""Alias for :func:`zeros_like`."""
rand = random
"""Alias for :func:`random`."""
randn = random
"""Alias for :func:`random`."""
randint = random
"""Alias for :func:`random`."""
