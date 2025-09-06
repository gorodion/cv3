# cv3: Pythonic OpenCV

cv3 is a Pythonic wrapper for OpenCV that simplifies computer vision tasks by providing
more intuitive interfaces and eliminating common boilerplate code.

## Principles
1. Solve more important tasks than writing extra code
2. You are programming in Python, not C++
3. OpenCV is not scary

## Why cv3?

OpenCV is a powerful computer vision library, but its API can be verbose and unintuitive,
especially for Python developers. cv3 addresses these issues by:

- Providing sensible defaults for common parameters
- Accepting more flexible input types (e.g., float coordinates, pathlib.Path objects)
- Eliminating repetitive code patterns
- Adding Pythonic features like context managers and iterators
- Handling common error cases with better error messages

## Key Benefits

- **Simplified API**: Common operations require fewer parameters
- **Type Flexibility**: Accepts various input types where OpenCV is strict
- **Error Handling**: Better error messages and automatic handling of common issues
- **Pythonic Design**: Context managers, iterators, and other Python idioms
- **RGB by Default**: Works with RGB color format by default (configurable)

## Installation

You can install cv3 using either of the following methods:

1. From PyPI (recommended):

```bash
pip install cv3
```

2. From GitHub (latest development version):

```bash
pip install git+https://github.com/gorodion/cv3.git
```

## Quick Start

```python
import cv3

# Read an image (automatically handles RGB conversion)
img = cv3.imread('image.jpg')

# Draw a rectangle without specifying color or thickness
cv3.rectangle(img, 100, 50, 150, 100)

# Save the image (automatically creates directories if needed)
cv3.imwrite('output/image_result.jpg', img, mkdir=True)

# Display in a window with context manager
with cv3.Window('Result') as window:
    window.imshow(img)
    window.wait_key(0)
```

This is just a small example of what cv3 can do. Check out the [documentation](https://cv3.readthedocs.io/en/latest/)
for a comprehensive overview of all the improvements and additions that cv3 provides over raw OpenCV.

You can also get acquainted with the features in [demo.ipynb](https://github.com/gorodion/pycv/blob/main/demo.ipynb)

## Run tests

```bash
pytest -v
```

---
I hope this is helpful, please contribute 🙂
