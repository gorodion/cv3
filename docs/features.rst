Features and Differences from cv2
=================================

cv3 provides numerous improvements and additions over raw OpenCV (cv2). Here's a detailed 
overview of the key features and how they differ from standard OpenCV usage.

Input/Output Operations
-----------------------

**Enhanced File Handling**
- Automatic RGB/BGR conversion (configurable via ``cv3.opt.RGB``)
- Better error handling with clear error messages
- Support for pathlib.Path objects
- Automatic directory creation when writing files (``mkdir=True``)

.. code-block:: python

   # cv3 - Simple and clear
   img = cv3.imread('image.jpg')  # Automatically handles RGB conversion
   cv3.imwrite('folder/output.jpg', img, mkdir=True)  # Creates folder automatically
   
   # cv2 - More verbose and error-prone
   img = cv2.imread('image.jpg')
   img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Manual conversion needed
   os.makedirs('folder', exist_ok=True)  # Manual directory creation
   cv2.imwrite('folder/output.jpg', img)

**Improved Video Handling**
- Context managers for automatic resource cleanup
- Iterator support for frames
- Direct access to frame count, current frame, FPS, dimensions
- Simplified VideoWriter initialization

.. code-block:: python

   # cv3 - Pythonic and clean
   with cv3.Video('input.mp4') as cap, cv3.Video('output.mp4', 'w') as out:
       for frame in cap:  # Iterator support
           processed = process_frame(frame)
           out.write(processed)
       print(f"Total frames: {len(cap)}")  # Direct access to frame count
       
   # cv2 - More boilerplate
   cap = cv2.VideoCapture('input.mp4')
   ret, frame = cap.read()
   while ret:
       processed = process_frame(frame)
       # ... write to output
       ret, frame = cap.read()
   cap.release()

Drawing Functions
-----------------

**Simplified Parameters**
- Sensible defaults for color, thickness, and other parameters
- Support for named colors ("red", "blue", etc.)
- Accepts float coordinates (automatically converted to integers)
- Relative coordinate support (``rel=True``)

.. code-block:: python

   # cv3 - Minimal parameters needed
   canvas = cv3.zeros(200, 200, 3)
   cv3.rectangle(canvas, 100, 50.3, 150.1, 100)  # Float coords accepted
   cv3.rectangle(canvas, 0.1, 0.2, 0.7, 0.7, rel=True)  # Relative coords
   cv3.rectangle(canvas, 10, 10, 50, 30, mode='xywh')  # Different coordinate modes
   
   # cv2 - Requires all parameters
   canvas = np.zeros((200, 200, 3), dtype=np.uint8)
   cv2.rectangle(canvas, (100, 50), (150, 100), (255, 255, 255), 1)

**Enhanced Functionality**
- ``copy=True`` parameter to avoid in-place modifications
- Multiple coordinate modes (xyxy, xywh, ccwh)
- Consistent parameter naming across functions

.. code-block:: python

   # cv3 - Non-destructive drawing
   original = cv3.zeros(200, 200, 3)
   with_square = cv3.rectangle(original, 100, 50, 150, 100, copy=True)
   # original remains unchanged
   
   # cv2 - Always modifies in-place
   original = np.zeros((200, 200, 3), dtype=np.uint8)
   with_square = cv2.rectangle(original, (100, 50), (150, 100), (255, 255, 255), 1)
   # original is now modified

Transformations
---------------

**Simplified Common Operations**
- Direct functions for common transformations (rotate90, rotate180, etc.)
- Combined operations (transform for simultaneous rotation and scaling)
- Support for relative coordinates
- Sensible default interpolation and border modes

.. code-block:: python

   # cv3 - Simple function calls
   rotated = cv3.rotate90(img)
   scaled = cv3.scale(img, 1.5)
   transformed = cv3.transform(img, angle=30, scale=0.8)
   
   # cv2 - More complex code
   (h, w) = img.shape[:2]
   center = (w // 2, h // 2)
   M = cv2.getRotationMatrix2D(center, 90, 1.0)
   rotated = cv2.warpAffine(img, M, (w, h))

Color Space Conversions
-----------------------

**Intuitive Function Names**
- Clear function names (rgb2gray, bgr2hsv, etc.)
- Automatic handling of RGB/BGR based on global setting
- Support for grayscale to color conversions

.. code-block:: python

   # cv3 - Clear and intuitive
   gray = cv3.rgb2gray(img)
   hsv = cv3.rgb2hsv(img)
   
   # cv2 - Requires remembering flag values
   gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
   hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

Configuration System
--------------------

**Global Options**
- Centralized configuration via ``cv3.opt``
- Easy customization of defaults (color, thickness, font, etc.)
- Module-specific settings (video FPS, codec)

.. code-block:: python

   # cv3 - Centralized configuration
   cv3.opt.COLOR = 'red'
   cv3.opt.THICKNESS = 2
   cv3.opt.video(fps=30, fourcc='mp4v')
   
   # cv2 - No centralized configuration
   # Need to pass parameters to each function call

Image Creation
--------------

**Convenient Factory Functions**
- Simple functions for creating images with common values
- Support for various data types (zeros, ones, random, etc.)

.. code-block:: python

   # cv3 - Simple creation functions
   black_img = cv3.zeros(100, 100, 3)
   white_img = cv3.white(100, 100, 3)
   random_img = cv3.random(100, 100, 3)
   
   # cv2 - More verbose
   black_img = np.zeros((100, 100, 3), dtype=np.uint8)
   white_img = np.full((100, 100, 3), 255, dtype=np.uint8)

Utility Functions
-----------------

**Coordinate Conversion**
- Functions for converting between coordinate formats
- Relative to absolute coordinate conversion
- Support for various bounding box formats

.. code-block:: python

   # cv3 - Built-in coordinate conversion
   x0, y0, x1, y1 = cv3.utils.xywh2xyxy(10, 20, 30, 40)
   x_rel, y_rel = cv3.utils.abs2rel(50, 75, width=100, height=100)
   
   # cv2 - No built-in support, need custom functions

Window Management
-----------------

**Enhanced Window Handling**
- Context managers for automatic cleanup
- Window classes for object-oriented interface
- Support for multiple windows

.. code-block:: python

   # cv3 - Context managers and classes
   with cv3.Window('Demo', pos=(100, 100)) as window:
       window.imshow(img)
       window.wait_key(0)
       
   with cv3.Windows(['Window1', 'Window2']) as windows:
       windows['Window1'].imshow(img1)
       windows['Window2'].imshow(img2)
       cv3.wait_key(0)
   
   # cv2 - Manual management
   cv2.namedWindow('Demo')
   cv2.moveWindow('Demo', 100, 100)
   cv2.imshow('Demo', img)
   cv2.waitKey(0)
   cv2.destroyWindow('Demo')