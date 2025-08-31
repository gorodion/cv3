"""cv3 - A simplified OpenCV wrapper library.

This package provides simplified interfaces for common OpenCV operations
including image creation, drawing, transformations, color space conversions,
and video processing.

Modules:
    opt: Global configuration options for the library.
    color_spaces: Functions for converting between color spaces.
    io: Input/output operations for images and videos.
    draw: Drawing functions for images.
    transform: Image transformation functions.
    processing: Basic image processing operations.
    video: Video capture and writing utilities.
    create: Functions for creating images with various initial values.
"""

from . import opt
from .color_spaces import *
from .io import *
from .draw import *
from .transform import *
from .processing import *
from .video import *
from .create import *
