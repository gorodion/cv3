"""Input/Output operations for image handling.

This module provides functions and classes for reading, writing, and displaying images
with automatic color space conversion and enhanced error handling.

Functions:
    is_ascii: Check if a string contains only ASCII characters.
    imdecode: Decode an image from a buffer.
    imread: Read an image from a file.
    imwrite: Write an image to a file.
    imshow: Display an image in a window.
    wait_key: Wait for a keyboard event.
    destroy_windows: Destroy all windows.
    destroy_window: Destroy a specific window.

Classes:
    Window: Manage a single display window.
    Windows: Manage multiple display windows.
"""
from itertools import cycle
from pathlib import Path
import warnings
import cv2
import numpy as np

from .color_spaces import rgb, rgba
from . import opt
from ._private._utils import typeit, type_decorator
from ._private._io import _imread_flag_match, _is_ascii

__all__ = [
    'imread',
    'imdecode',
    'imwrite',
    'imshow',
    'Window',
    'Windows',
    'wait_key', 'waitKey',
    'destroy_windows', 'destroyAllWindows',
    'destroy_window', 'destroyWindow'
]


def is_ascii(s):
    """Check if a string contains only ASCII characters.

    Args:
        s (str): String to check.

    Returns:
        bool: True if all characters in the string are ASCII, False otherwise.
    """
    return _is_ascii(s)

def imdecode(buf, flag):
    """Decode an image from a buffer.

    Args:
        buf (numpy.ndarray): Buffer containing the image data.
        flag (int or str): Flag specifying the color type of the decoded image.
            Can be one of: 'color', 'gray', 'alpha', 'unchanged' or OpenCV flags.

    Returns:
        numpy.ndarray: Decoded image.

    Example:
        >>> import cv3
        >>> import numpy as np
        >>> # Read image file as bytes
        >>> with open('image.jpg', 'rb') as f:
        ...     buf = np.frombuffer(f.read(), dtype=np.uint8)
        >>> # Decode image
        >>> img = cv3.imdecode(buf, 'color')
    """
    if isinstance(flag, str):
        flag = _imread_flag_match(flag)
    img = cv2.imdecode(buf, flag)
    return img

def imread(img_path, flag=cv2.IMREAD_COLOR):
    """Read an image from a file.

    Args:
        img_path (str or Path): Path to the image file.
        flag (int or str, optional): Flag specifying the color type of the loaded image.
            Can be one of: 'color', 'gray', 'alpha', 'unchanged' or OpenCV flags.
            Defaults to cv2.IMREAD_COLOR.

    Returns:
        numpy.ndarray: Loaded image.

    Raises:
        IsADirectoryError: If img_path is a directory.
        FileNotFoundError: If img_path does not exist.
        OSError: If the image file cannot be read.

    Note:
        This function automatically handles RGB/BGR color space conversion based on
        the opt.RGB setting. When opt.RGB is True (default), the image is converted
        from BGR to RGB.

    Example:
        >>> import cv3
        >>> # Read a color image
        >>> img = cv3.imread('image.jpg')
        >>> # Read a grayscale image
        >>> img = cv3.imread('image.jpg', 'gray')
        >>> # Read an image with alpha channel
        >>> img = cv3.imread('image.png', 'unchanged')
    """
    if Path(img_path).is_dir():
        raise IsADirectoryError(str(img_path))
    if not Path(img_path).is_file():
        raise FileNotFoundError(str(img_path))
    if isinstance(img_path, Path):
        img_path = str(img_path)
    if isinstance(flag, str):
        flag = _imread_flag_match(flag)
    img = cv2.imread(img_path, flag)
    if img is None:
        if not _is_ascii(img_path):
            img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), flag)
        if img is None:
            raise OSError('File was not read: {}'.format(img_path))
    if img.ndim == 2:
        return img
    if opt.RGB:
        if img.shape[-1] == 4:
            img = rgba(img)
        else:
            img = rgb(img)
    return img


def imwrite(img_path, img, mkdir=False, ascii=True):
    """Write an image to a file.

    Args:
        img_path (str or Path): Path to the image file to write.
        img (numpy.ndarray): Image to write.
        mkdir (bool, optional): If True, create parent directories if they don't exist.
            Defaults to False.
        ascii (bool, optional): If True, use ASCII filename handling. If False, use
            non-ASCII filename handling with numpy's tofile method. Defaults to True.

    Raises:
        OSError: If the image cannot be written to the file.

    Note:
        This function automatically handles RGB/BGR color space conversion based on
        the opt.RGB setting. When opt.RGB is True (default), the image is converted
        from RGB to BGR before writing.

        When ascii=False, the function uses numpy's tofile method to handle non-ASCII
        filenames, which is useful for international characters in filenames.

    Example:
        >>> import cv3
        >>> import numpy as np
        >>> # Create a simple image
        >>> img = np.zeros((100, 100, 3), dtype=np.uint8)
        >>> img[:, :] = [255, 0, 0]  # Blue image (BGR)
        >>> # Write image to file
        >>> cv3.imwrite('output.jpg', img)
        >>> # Write image with automatic directory creation
        >>> cv3.imwrite('output_dir/output.jpg', img, mkdir=True)
    """
    if mkdir:
        Path(img_path).parent.mkdir(parents=True, exist_ok=True)
    if isinstance(img_path, Path):
        img_path = str(img_path)
    if opt.RGB:
        img = rgb(img)  # includes typeit
    else:
        img = typeit(img)

    if not ascii:
        # if is_ascii(img_path):
        #     warnings.warn('Passed ascii filename but `ascii`=True')
        ext = Path(img_path).suffix
        ret2, buf = cv2.imencode(ext=ext, img=img)
        if not ret2:
            raise OSError('Something went wrong when writing image (non-ascii filename)')
        buf.tofile(img_path)
        return

    ret = cv2.imwrite(img_path, img)
    if not ret:
        raise OSError('Something went wrong when writing image')


def imshow(window_name, img):
    """Display an image in a window.

    Args:
        window_name (str): Name of the window to display the image in.
        img (numpy.ndarray): Image to display.

    Note:
        This function automatically handles RGB/BGR color space conversion based on
        the opt.RGB setting. When opt.RGB is True (default), the image is converted
        from RGB to BGR before displaying.

    Example:
        >>> import cv3
        >>> import numpy as np
        >>> # Create a simple image
        >>> img = np.zeros((100, 100, 3), dtype=np.uint8)
        >>> img[:, :] = [255, 0, 0]  # Blue image (BGR)
        >>> # Display the image
        >>> cv3.imshow('My Image', img)
        >>> cv3.waitKey(0)
    """
    if opt.RGB:
        img = rgb(img)
    else:
        img = typeit(img)
    cv2.imshow(window_name, img)


def wait_key(t):
    """Wait for a keyboard event.

    Args:
        t (int): Delay in milliseconds. If 0, it waits indefinitely for a key stroke.

    Returns:
        int: The code of the pressed key, or -1 if no key was pressed before the timeout.

    Note:
        This function is a wrapper around cv2.waitKey() that masks the return value
        with 0xFF to ensure consistent behavior across different platforms.

    Example:
        >>> import cv3
        >>> # Wait for a key press for 1 second
        >>> key = cv3.wait_key(1000)
        >>> if key != -1:
        ...     print(f'Key pressed: {key}')
    """
    return cv2.waitKey(t) & 0xFF

class Window:
    """A class to manage a single display window.
    
    This class provides a convenient way to create and manage OpenCV windows
    with additional features like automatic naming and context manager support.
    
    Attributes:
        window_name (str): The name of the window.
    """
    __window_count = 0

    def __init__(self, window_name=None, pos=None, flag=cv2.WINDOW_AUTOSIZE):
        """Initialize a Window object.

        Args:
            window_name (str, optional): Name of the window. If None, a name will be
                automatically generated. Defaults to None.
            pos (tuple, optional): Starting position of the window as (x, y).
                Defaults to None.
            flag (int, optional): Window flag. Defaults to cv2.WINDOW_AUTOSIZE.

        Example:
            >>> import cv3
            >>> import numpy as np
            >>> # Create a window with automatic naming
            >>> window = cv3.Window()
            >>> # Create a window with a specific name and position
            >>> window = cv3.Window('My Window', pos=(100, 100))
        """
        if window_name is None:
            window_name = 'window{}'.format(Window.__window_count)

        window_name = str(window_name)
        cv2.namedWindow(window_name, flag)

        if pos is not None:
            cv2.moveWindow(window_name, *pos)

        self.window_name = window_name
        Window.__window_count += 1

    def imshow(self, img):
        """Display an image in this window.

        Args:
            img (numpy.ndarray): Image to display.

        Note:
            This function automatically handles RGB/BGR color space conversion based on
            the opt.RGB setting. When opt.RGB is True (default), the image is converted
            from RGB to BGR before displaying.

        Example:
            >>> import cv3
            >>> import numpy as np
            >>> # Create a window
            >>> window = cv3.Window('My Window')
            >>> # Create an image
            >>> img = np.zeros((100, 100, 3), dtype=np.uint8)
            >>> img[:, :] = [255, 0, 0]  # Blue image (BGR)
            >>> # Display the image
            >>> window.imshow(img)
        """
        if opt.RGB:
            img = rgb(img)
        else:
            img = typeit(img)
        cv2.imshow(self.window_name, img)

    def move(self, x, y):
        """Move the window to a new position.

        Args:
            x (int): New x-coordinate of the window.
            y (int): New y-coordinate of the window.

        Example:
            >>> import cv3
            >>> # Create a window
            >>> window = cv3.Window('My Window')
            >>> # Move the window to position (200, 200)
            >>> window.move(200, 200)
        """
        cv2.moveWindow(self.window_name, x, y)

    def close(self):
        """Close this window and free associated resources.

        Example:
            >>> import cv3
            >>> # Create a window
            >>> window = cv3.Window('My Window')
            >>> # Close the window
            >>> window.close()
        """
        cv2.destroyWindow(self.window_name)

    @staticmethod
    def wait_key(t):
        """Wait for a keyboard event.

        This is a static method that calls the module-level wait_key function.

        Args:
            t (int): Delay in milliseconds. If 0, it waits indefinitely for a key stroke.

        Returns:
            int: The code of the pressed key, or -1 if no key was pressed before the timeout.

        Example:
            >>> import cv3
            >>> # Create a window
            >>> window = cv3.Window('My Window')
            >>> # Wait for a key press for 1 second
            >>> key = window.wait_key(1000)
        """
        return wait_key(t)

    def __enter__(self):
        """Enter the runtime context for the window.
        
        Returns:
            Window: This window instance.
            
        Example:
            >>> import cv3
            >>> # Use window as a context manager
            >>> with cv3.Window('My Window') as window:
            ...     # Window is automatically closed when exiting the context
            ...     pass
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the runtime context for the window.
        
        This method ensures the window is closed when exiting the context manager.
        """
        self.close()


class Windows:
    """A class to manage multiple display windows.
    
    This class provides a convenient way to create and manage multiple OpenCV windows
    with additional features like context manager support.
    
    Attributes:
        windows (dict): A dictionary mapping window names to Window objects.
    """
    
    def __init__(self, window_names, poses=None):
        """Initialize a Windows object.

        Args:
            window_names (list): List of window names.
            poses (list, optional): List of positions for each window as (x, y) tuples.
                If None, windows will use default positions. Defaults to None.

        Example:
            >>> import cv3
            >>> # Create multiple windows
            >>> windows = cv3.Windows(['Window1', 'Window2'])
            >>> # Create multiple windows with specific positions
            >>> windows = cv3.Windows(['Window1', 'Window2'], poses=[(100, 100), (200, 200)])
        """
        if poses is None:
            poses = (None,) * len(window_names)

        self.windows = {}
        for window_name, pos in zip(window_names, poses):
            self.windows[window_name] = Window(window_name, pos=pos)

    def __getitem__(self, name):
        """Get a specific window by name.

        Args:
            name (str): Name of the window to retrieve.

        Returns:
            Window: The requested window object.

        Example:
            >>> import cv3
            >>> # Create multiple windows
            >>> windows = cv3.Windows(['Window1', 'Window2'])
            >>> # Access a specific window
            >>> window1 = windows['Window1']
        """
        return self.windows[name]

    def close(self):
        """Close all windows and free associated resources.

        Example:
            >>> import cv3
            >>> # Create multiple windows
            >>> windows = cv3.Windows(['Window1', 'Window2'])
            >>> # Close all windows
            >>> windows.close()
        """
        for window_name in self.windows:
            self.windows[window_name].close()

    @staticmethod
    def wait_key(t):
        """Wait for a keyboard event.

        This is a static method that calls the module-level wait_key function.

        Args:
            t (int): Delay in milliseconds. If 0, it waits indefinitely for a key stroke.

        Returns:
            int: The code of the pressed key, or -1 if no key was pressed before the timeout.

        Example:
            >>> import cv3
            >>> # Create multiple windows
            >>> windows = cv3.Windows(['Window1', 'Window2'])
            >>> # Wait for a key press for 1 second
            >>> key = windows.wait_key(1000)
        """
        return wait_key(t)

    def __enter__(self):
        """Enter the runtime context for the windows.
        
        Returns:
            Windows: This windows instance.
            
        Example:
            >>> import cv3
            >>> # Use windows as a context manager
            >>> with cv3.Windows(['Window1', 'Window2']) as windows:
            ...     # All windows are automatically closed when exiting the context
            ...     pass
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the runtime context for the windows.
        
        This method ensures all windows are closed when exiting the context manager.
        """
        self.close()

def destroy_windows():
    """Destroy all windows.
    
    This function destroys all windows that were created with imshow or Window.
    
    Note:
        This is a wrapper around cv2.destroyAllWindows().
        
    Example:
        >>> import cv3
        >>> # Create and display an image
        >>> import numpy as np
        >>> img = np.zeros((100, 100, 3), dtype=np.uint8)
        >>> cv3.imshow('Window1', img)
        >>> cv3.imshow('Window2', img)
        >>> # Destroy all windows
        >>> cv3.destroy_windows()
    """
    cv2.destroyAllWindows()


def destroy_window(winname: str):
    """Destroy a specific window.
    
    Args:
        winname (str): Name of the window to destroy.
        
    Note:
        This is a wrapper around cv2.destroyWindow().
        
    Example:
        >>> import cv3
        >>> # Create and display an image
        >>> import numpy as np
        >>> img = np.zeros((100, 100, 3), dtype=np.uint8)
        >>> cv3.imshow('My Window', img)
        >>> # Destroy the specific window
        >>> cv3.destroy_window('My Window')
    """
    cv2.destroyWindow(winname)

# Aliases
waitKey = wait_key
"""Alias for :func:`wait_key`."""
destroyAllWindows = destroy_windows
"""Alias for :func:`destroy_windows`."""
destroyWindow = destroy_window
"""Alias for :func:`destroy_window`."""

