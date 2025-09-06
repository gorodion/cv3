Core Concepts and Package Features
==================================

This page provides a detailed explanation of the core concepts and distinctive features that make cv3 a powerful and user-friendly wrapper for OpenCV.

RGB Format by Default
---------------------

cv3 uses RGB format by default for images, which is more intuitive for most Python developers. This behavior can be changed globally using ``cv3.opt.RGB``. The global RGB setting affects all modules: image/video reading/writing, drawing functions, and other operations.

.. code-block:: python

   import cv3
   
   # By default, cv3 reads images in RGB mode (unlike OpenCV which uses BGR)
   img = cv3.imread('image.jpg')  # Read in RGB format by default
   print(cv3.opt.RGB)  # True (default)
   
   # You can switch to BGR mode globally
   cv3.opt.set_bgr()  # Sets RGB=False
   print(cv3.opt.RGB)  # False
   img_bgr = cv3.imread('image.jpg')  # Now reads in BGR format
   
   # Or switch back to RGB mode
   cv3.opt.set_rgb()  # Sets RGB=True
   print(cv3.opt.RGB)  # True

Auto Scaling with Relative Coordinates
--------------------------------------

Many cv3 functions support relative coordinates through the ``rel=True`` parameter, making it easier to work with images of different sizes. This feature is available in both drawing and transformation functions.

Drawing with Relative Coordinates
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import cv3
   
   img = cv3.zeros(400, 600, 3)  # 600x400 image
   
   # Draw a rectangle at 10% from left, 20% from top, 
   # with width 30% of image width and height 25% of image height
   cv3.rectangle(img, 0.1, 0.2, 0.4, 0.45, rel=True)
   
   # Draw a circle at center of image with radius 10% of image width
   cv3.circle(img, 0.5, 0.5, 0.1, rel=True)

Transformations with Relative Coordinates
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import cv3
   
   img = cv3.imread('image.jpg')
   
   # Shift image by 10% of width to the right and 5% of height down
   shifted = cv3.shift(img, 0.1, 0.05, rel=True)
   
   # Pad image with 5% of width on left/right and 3% of height on top/bottom
   padded = cv3.pad(img, 0.03, 0.03, 0.05, 0.05, rel=True)
   
   # Crop with relative coordinates
   cropped = cv3.crop(img, 0.25, 0.25, 0.75, 0.75, rel=True)

Combined Scale and Rotate with cv3.transform
--------------------------------------------

The ``cv3.transform`` function provides a convenient way to apply both rotation and scaling in a single operation, which is more efficient than applying them separately.

.. code-block:: python

   import cv3
   
   img = cv3.imread('image.jpg')
   
   # Rotate by 30 degrees and scale by 1.5 in one operation
   transformed = cv3.transform(img, angle=30, scale=1.5)
   
   # Compare with cv2 approach which requires more steps
   # (h, w) = img.shape[:2]
   # center = (w // 2, h // 2)
   # M = cv2.getRotationMatrix2D(center, 30, 1.5)
   # transformed = cv2.warpAffine(img, M, (w, h))

Dynamic Typing Support
----------------------

cv3 functions are designed to be flexible with input types, automatically handling conversions where appropriate. This includes supporting non-uint8 image formats and floating-point coordinates.

Handling Non-uint8 Image Formats
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import cv3
   import numpy as np
   
   # Create a float image (not uint8)
   float_img = np.random.rand(100, 100, 3).astype(np.float32)
   
   # cv3 functions automatically handle the type conversion
   cropped = cv3.crop(float_img, 10, 10, 50, 50)
   print(cropped.dtype)  # float32 (preserved)
   
   # Drawing on float images also works
   cv3.rectangle(float_img, 20, 20, 80, 80, color=(1.0, 0.0, 0.0))

Handling Floating-Point Coordinates
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import cv3
   
   img = cv3.zeros(100, 100, 3)
   
   # Float coordinates are automatically converted to integers
   cv3.rectangle(img, 10.7, 20.3, 50.9, 80.1)  # Works fine
   cropped = cv3.crop(img, 25.5, 30.8, 75.2, 80.9)  # Also works fine

Unified Video Interface
-----------------------

cv3 provides a consistent interface for both video reading and writing through the ``cv3.Video`` class, with support for context managers and convenient properties.

.. code-block:: python

   import cv3
   
   # Reading video with context manager
   with cv3.Video('input.mp4') as reader:
       print(f"Video dimensions: {reader.width}x{reader.height}")
       print(f"Frame rate: {reader.fps}")
       print(f"Total frames: {len(reader)}")
       
       # Iterate through frames
       for i, frame in enumerate(cv3.Video('input.mp4')):
           if i < 5:  # Process first 5 frames
               processed = process_frame(frame)
               # Do something with processed frame
       # width and height are available after initialization
       print(f"Video dimensions: {reader.width}x{reader.height}")
       print(f"Frame rate: {reader.fps}")

   # Writing video with context manager
   with cv3.Video('output.mp4', 'w', fps=30) as writer:
       # Write frames
       for frame in cv3.Video('input.mp4'):
           writer.write(frame)
       
       print(f"Output dimensions: {writer.width}x{writer.height}")
       print(f"Frame rate: {writer.fps}")
       

   # You can also use it without context manager
   reader = cv3.Video('input.mp4')
   writer = cv3.Video('output.mp4', 'w')
   
   # Don't forget to release resources manually when not using context managers
   reader.close() # or reader.release()
   writer.close() # or writer.release()

Video Reading Features
^^^^^^^^^^^^^^^^^^^^^^^

cv3.Video provides convenient features for video reading, including list comprehension
for processing all frames, frame indexing, and current frame access.

.. code-block:: python

   import cv3
   
   with cv3.Video('input.mp4') as cap:
       # Process all frames with list comprehension
       frames = [cv3.resize(frame, 10, 10) for frame in cap]
       
       # Rewind the video
       cap.rewind(0)  # or cap.seek(0)

       frame = cap.read()

       # Get current frame index (it's 1)
       current_frame = cap.now

       # Access a specific frame by index
       frame = cap[100]  # Get frame at index 100
       
       # Get current frame index (it's 101)
       current_frame = cap.now
       
       # Get total number of frames
       total_frames = len(cap)

File Path Handling and Automatic Directory Creation
----------------------------------------------------

cv3 functions accept ``pathlib.Path`` objects in addition to string paths, making it easier to work with modern Python path handling. When writing files, cv3 can automatically create directories if they don't exist using the ``mkdir=True`` parameter.

.. code-block:: python

   import cv3
   from pathlib import Path
   
   # Using pathlib.Path objects
   image_path = Path('images') / 'photo.jpg'
   img = cv3.imread(image_path)
   
   # Save to a pathlib.Path with automatic directory creation
   output_path = Path('output') / 'result.jpg'
   cv3.imwrite(output_path, img, mkdir=True)  # Automatically creates directories
   
   img = cv3.zeros(100, 100, 3)
   
   # This will automatically create the 'output/nested' directory structure
   cv3.imwrite('output/nested/result.jpg', img, mkdir=True)
   
   # For videos too
   with cv3.Video('output/video/result.mp4', 'w', mkdir=True) as writer:
       writer.write(img)

Window Interface with Context Manager
-------------------------------------

cv3 provides a clean window interface with context managers for automatic resource cleanup.

.. code-block:: python

   import cv3
   
   img = cv3.zeros(200, 200, 3)
   cv3.rectangle(img, 50, 50, 150, 150)
   
   # Using context manager for automatic cleanup
   with cv3.Window('Demo Window', pos=(100, 100)) as window:
       window.imshow(img)
       window.wait_key(0)  # Press any key to continue
       
   # Multiple windows with context manager
   with cv3.Windows(['Window1', 'Window2']) as windows:
       windows['Window1'].imshow(img)
       windows['Window2'].imshow(img)
       cv3.wait_key(0)  # Press any key to close all windows
       
   # Display video from webcam
   with cv3.Window('Webcam') as window:
       for frame in cv3.Video(0):
           window.imshow(frame)
           if cv3.wait_key(1) == ord('q'):
               break

Coordinate Format Flexibility
-----------------------------

cv3 drawing and cropping functions support multiple coordinate formats, making it easier to work with different data sources.

.. code-block:: python

   import cv3
   
   img = cv3.zeros(100, 100, 3)
   
   # Default mode is 'xyxy' (x0, y0, x1, y1)
   cv3.rectangle(img, 10, 10, 50, 50)
   
   # 'xywh' mode (x, y, width, height)
   cv3.rectangle(img, 60, 10, 30, 30, mode='xywh')
   
   # 'ccwh' mode (center_x, center_y, width, height)
   cv3.rectangle(img, 25, 75, 30, 30, mode='ccwh')
   
   # Cropping with different modes
   cropped1 = cv3.crop(img, 10, 10, 50, 50)  # xyxy
   cropped2 = cv3.crop(img, 10, 10, 40, 40, mode='xywh')  # xywh

Drawing Functions
-----------------

cv3 provides a comprehensive set of drawing functions with enhanced features compared to OpenCV. These functions support relative coordinates, flexible color handling, and various coordinate formats.

cv3 provides standard drawing functions like rectangle, circle, text, polylines, fill_poly, etc.

.. code-block:: python

   import cv3
   
   # Example with poly functions
   pts = [
       [30, 30],
       [160, 30],
       [45, 150]
   ]
   
   canvas = cv3.zeros(200, 200, 3)
   cv3.rectangle(canvas, 10, 10, 100, 100, color='blue')
   cv3.circle(canvas, 150, 50, 30, color='red')
   cv3.text(canvas, "Hello", 50, 150, color='green')
   cv3.polylines(canvas, pts, is_closed=True, color='red', t=5)
   cv3.fill_poly(canvas, pts, color='yellow')

Drawing with Relative Coordinates
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Many cv3 drawing functions support relative coordinates through the ``rel=True`` parameter.

.. code-block:: python

   import cv3
   
   img = cv3.zeros(400, 600, 3)  # 600x400 image
   
   # Draw a rectangle at 10% from left, 20% from top,
   # with width 30% of image width and height 25% of image height
   cv3.rectangle(img, 0.1, 0.2, 0.4, 0.45, rel=True)
   
   # Draw text at 50% from left, 80% from top
   cv3.text(img, "Hello World", 0.5, 0.8, rel=True, font_scale=1.5)

Fill Parameter
^^^^^^^^^^^^^^

The ``fill`` parameter allows you to create filled shapes.

.. code-block:: python

   import cv3
   
   img = cv3.zeros(200, 200, 3)
   
   # Filled rectangle
   cv3.rectangle(img, 20, 20, 80, 80, fill=True, color='red')
   
   # Filled circle
   cv3.circle(img, 150, 50, 30, fill=True, color='blue')

Non-Destructive Drawing
^^^^^^^^^^^^^^^^^^^^^^^

By default, cv3 drawing functions modify images in-place, but you can use the ``copy=True`` parameter to avoid modifying the original image.

.. code-block:: python

   import cv3
   
   img = cv3.zeros(100, 100, 3)
   
   # Default behavior: modifies original image
   cv3.rectangle(img, 10, 10, 50, 50)  # img is now modified
   
   # Non-destructive drawing: creates a copy
   img2 = cv3.zeros(100, 100, 3)
   result = cv3.rectangle(img2, 10, 10, 50, 50, copy=True)
   # img2 is unchanged, result contains the drawing

Color Handling
^^^^^^^^^^^^^^

cv3 provides flexible color handling in drawing functions. Available named colors can be found in ``cv3.COLORS``.

.. code-block:: python

   import cv3
   
   img = cv3.zeros(100, 100, 3)
   
   # Named colors
   cv3.rectangle(img, 10, 10, 50, 50, color='red')
   cv3.circle(img, 75, 25, 10, color='blue')
   
   # RGB tuples
   cv3.rectangle(img, 10, 60, 50, 90, color=(0, 255, 0))  # Green
   
   # Single value for grayscale
   gray_img = cv3.zeros(100, 100)
   cv3.rectangle(gray_img, 10, 10, 50, 50, color=128)  # Gray

Border Colors in Transformations
""""""""""""""""""""""""""""""""

cv3 also provides flexible color handling in transformations that require border values.

.. code-block:: python

   import cv3
   
   img = cv3.zeros(100, 100, 3)
   
   # Shift with border color
   shifted = cv3.shift(img, 20, 10, value=(255, 0, 0))  # Red border
   
   # Pad with border color
   padded = cv3.pad(img, 10, 10, 10, 10, value='green')

Clear Error Messages
--------------------

cv3 provides more informative error messages compared to raw OpenCV, making it easier to debug issues.

.. code-block:: python

   import cv3
   
   # Trying to read a non-existent file
   try:
       img = cv3.imread('non_existent.jpg')
   except FileNotFoundError as e:
       print(f"Clear error message: {e}")
   
   # Trying to write to an invalid path without mkdir
   try:
       img = cv3.zeros(100, 100, 3)
       cv3.imwrite('path/that/does/not/exist.jpg', img)
   except FileNotFoundError as e:
       print(f"Clear error message: {e}")
       
   # OSError when file exists but cannot be read
   try:
       img = cv3.imread('/path/to/directory')  # Trying to read a directory
   except OSError as e:
       print(f"Clear error message: {e}")
       
   # StopIteration when video stream ends
   try:
       with cv3.Video('input.mp4') as cap:
           while True:
               frame = cap.read()
   except StopIteration as exc:
       print(f"Video stream ended: {exc}")

Convenient Color Space Conversions
----------------------------------

cv3 provides intuitive functions for color space conversions with clear naming conventions.

.. code-block:: python

   import cv3
   
   img = cv3.zeros(100, 100, 3)  # RGB image
   
   # Simple conversions with clear names
   gray = cv3.rgb2gray(img)
   hsv = cv3.rgb2hsv(img)
   bgr = cv3.rgb2bgr(img)
   
   # Reverse conversions
   rgb_from_gray = cv3.gray2rgb(gray)
   rgb_from_hsv = cv3.hsv2rgb(hsv)
   rgb_from_bgr = cv3.bgr2rgb(bgr)
   
Global Options System
---------------------

cv3 provides a centralized configuration system through ``cv3.opt`` that allows you to set default values for various parameters.

.. code-block:: python

  import cv3
  
  # Set global defaults for drawing functions
  cv3.opt.COLOR = 'red'  # Default color for drawing
  cv3.opt.THICKNESS = 2  # Default thickness
  cv3.opt.FONT = cv2.FONT_HERSHEY_SIMPLEX  # Default font
  
  # Set global defaults for video
  cv3.opt.video(fps=30, fourcc='mp4v')
  
  # Set global defaults for drawing
  cv3.opt.draw(thickness=3, color='blue')
  
  # Now all drawing functions will use these defaults
  img = cv3.zeros(100, 100, 3)
  cv3.rectangle(img, 10, 10, 90, 90)  # Uses red color and thickness 2

Additional Functions
--------------------

cv3 provides utility functions for common operations in the create and utils modules.

Image Creation Functions
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

  import cv3
  
  # Create images with different properties
  black_img = cv3.zeros(100, 100, 3)  # Black RGB image
  white_img = cv3.white(100, 100, 3)  # White RGB image
  random_img = cv3.random(100, 100, 3)  # Random noise image
  filled_img = cv3.full(100, 100, 3, value=(255, 128, 0))  # Orange image

Utility Functions
^^^^^^^^^^^^^^^^^

.. code-block:: python

  import cv3
  
  # Coordinate conversion utilities
  x0, y0, x1, y1 = cv3.utils.xywh2xyxy(10, 20, 30, 40)
  x_rel, y_rel = cv3.utils.abs2rel(50, 75, width=100, height=100)