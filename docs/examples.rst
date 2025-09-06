Examples
========

Here are practical examples demonstrating how to use cv3 for common computer vision tasks.

Basic Image Operations
----------------------

Reading, processing, and saving an image:

.. code-block:: python

   import cv3
   
   # Read an image
   img = cv3.imread('input.jpg')
   
   # Convert to grayscale
   gray = cv3.rgb2gray(img)
   
   # Apply threshold
   thresh = cv3.threshold(gray, thr=127, max=255)
   
   # Save the result
   cv3.imwrite('output_threshold.jpg', thresh, mkdir=True)
   
   # Display the result
   with cv3.Window('Threshold Result') as window:
       window.imshow(thresh)
       window.wait_key(0)

Drawing on Images
-----------------

Creating annotations and visualizations:

.. code-block:: python

   import cv3
   
   # Create a blank canvas
   canvas = cv3.zeros(400, 400, 3)
   
   # Draw various shapes
   cv3.rectangle(canvas, 50, 50, 200, 150, color='red', t=3)
   cv3.circle(canvas, 300, 100, 50, color='blue', t=-1)  # Filled circle
   cv3.line(canvas, 0, 0, 400, 400, color='green', t=2)
   
   # Add text
   cv3.text(canvas, 'Hello cv3!', 100, 300, color='white', scale=1.5)
   
   # Display
   with cv3.Window('Drawing Example') as window:
       window.imshow(canvas)
       window.wait_key(0)

Video Processing and Display
----------------------------

Processing video frames:

.. code-block:: python

   import cv3
   
   # Process video frames
   with (
       cv3.Video('input.mp4') as cap,
       cv3.Video('output.mp4', 'w') as out,
       cv3.Windows(['Original', 'Processed']) as windows
   ):
       for frame in cap:
           # Convert to grayscale
           gray = cv3.rgb2gray(frame)
           
           # Apply threshold
           thresh = cv3.threshold(gray)
           
           # Write processed frame
           out.write(thresh)
           
           # Display original and processed frames
           windows['Original'].imshow(frame)
           windows['Processed'].imshow(thresh)
               
           if cv3.wait_key(1) == ord('q'):
               break

Advanced Drawing with Coordinates
---------------------------------

Using different coordinate modes and relative coordinates:

.. code-block:: python

   import cv3
   
   img = cv3.imread('image.jpg')
   
   # Draw rectangles using different coordinate modes
   # Standard mode (x0, y0, x1, y1)
   cv3.rectangle(img, 10, 10, 100, 100, color='red', t=2)
   
   # Width/height mode (x, y, width, height) with fill parameter
   cv3.rectangle(img, 120, 10, 100, 100, mode='xywh', color='blue', t=2, fill=True)
   
   # Center/width/height mode (center_x, center_y, width, height)
   cv3.rectangle(img, 250, 50, 100, 100, mode='ccwh', color='green', t=2)
   
   # Using relative coordinates (0-1 range)
   cv3.rectangle(img, 0.7, 0.1, 0.9, 0.3, rel=True, color='yellow', t=2)
   
   # Display result
   with cv3.Window('Coordinate Modes') as window:
       window.imshow(img)
       window.wait_key(0)

Image Transformations
---------------------

Applying various transformations to images:

.. code-block:: python

   import cv3
   
   img = cv3.imread('image.jpg')
   
   # Rotate by 45 degrees
   rotated = cv3.rotate(img, 45)
   
   # Scale by 1.5x
   scaled = cv3.scale(img, 1.5)
   
   # Rotate and scale simultaneously
   transformed = cv3.transform(img, angle=30, scale=0.8)
   
   # Flip operations
   hflipped = cv3.hflip(img)  # Horizontal flip
   vflipped = cv3.vflip(img)  # Vertical flip
   dflipped = cv3.dflip(img)  # Diagonal flip
   
   # Display results
   with cv3.Windows(['Original', 'Rotated', 'Scaled', 'Transformed']) as windows:
       windows['Original'].imshow(img)
       windows['Rotated'].imshow(rotated)
       windows['Scaled'].imshow(scaled)
       windows['Transformed'].imshow(transformed)
       cv3.wait_key(0)

Color Space Manipulation
------------------------

Working with different color spaces:

.. code-block:: python

   import cv3
   
   img = cv3.imread('image.jpg')
   
   # Convert between color spaces
   gray = cv3.rgb2gray(img)
   hsv = cv3.rgb2hsv(img)
   
   # Convert back to RGB
   rgb_from_hsv = cv3.hsv2rgb(hsv)
   
   # Display results
   with cv3.Windows(['Original', 'Grayscale', 'HSV', 'HSV->RGB']) as windows:
       windows['Original'].imshow(img)
       windows['Grayscale'].imshow(gray)
       windows['HSV'].imshow(hsv)
       windows['HSV->RGB'].imshow(rgb_from_hsv)
       cv3.wait_key(0)

Creating Image Mosaics
----------------------

Combining multiple operations:

.. code-block:: python

   import cv3
   import numpy as np
   
   # Create a complex visualization
   canvas = cv3.white(600, 800, 3)
   
   # Draw a grid
   for i in range(0, 800, 50):
       cv3.vline(canvas, i, color='lightgray', t=1)
   for i in range(0, 600, 50):
       cv3.hline(canvas, i, color='lightgray', t=1)
   
   # Draw shapes with different colors
   shapes = [
       ('Rectangle', lambda x, y: cv3.rectangle(canvas, x, y, x+80, y+60, color='red', t=2)),
       ('Circle', lambda x, y: cv3.circle(canvas, x+40, y+30, 30, color='blue', t=2)),
       ('Triangle', lambda x, y: cv3.polylines(canvas, [[x, y+60], [x+40, y], [x+80, y+60]], 
                                               is_closed=True, color='green', t=2)),
   ]
   
   # Place shapes in a grid
   for i, (name, draw_func) in enumerate(shapes):
       x = 100 + (i % 3) * 200
       y = 100 + (i // 3) * 150
       draw_func(x, y)
       cv3.text(canvas, name, x, y-20, color='black', scale=0.7)
   
   # Display result
   with cv3.Window('Image Mosaic') as window:
       window.imshow(canvas)
       window.wait_key(0)

Configuration Customization
---------------------------

Customizing default behavior:

.. code-block:: python

   import cv3
   
   # Customize default drawing parameters
   cv3.opt.COLOR = 'red'
   cv3.opt.THICKNESS = 3
   cv3.opt.FONT = cv2.FONT_HERSHEY_SIMPLEX
   cv3.opt.SCALE = 1.2
   
   # Now all drawing functions use these defaults
   canvas = cv3.zeros(300, 300, 3)
   cv3.rectangle(canvas, 50, 50, 250, 250)  # Uses red color and thickness 3
   cv3.text(canvas, 'Custom Defaults', 60, 150)  # Uses custom font and scale
   
   # Customize video parameters
   cv3.opt.video(fps=30, fourcc='mp4v')
   
   # Create video with custom defaults
   with cv3.Video('custom_video.mp4', 'w') as out:
       frame = cv3.zeros(480, 640, 3)
       
       cv3.rectangle(frame, 100, 100, 500, 400, color='blue', t=5)
       cv3.text(frame, 'Custom Video', 200, 300, color='white')
       
       for i in range(100):
           out.write(frame)
   
   # Display result
   with cv3.Window('Custom Configuration') as window:
       window.imshow(canvas)
       window.wait_key(0)