"""Video processing operations.

This module provides classes and functions for reading and writing video files
with automatic color space conversion and enhanced error handling.

Classes:
    VideoCapture: Read video files or capture from camera devices.
    VideoWriter: Write video files with customizable encoding parameters.
    VideoInterface: Base class for video operations.

Functions:
    Video: Factory function to create VideoCapture or VideoWriter instances.

Aliases:
    VideoReader: Alias for VideoCapture class.
"""
import warnings
from pathlib import Path
import numpy as np
import cv2
from cv2 import VideoCapture as BaseVideoCapture, VideoWriter as BaseVideoWriter
from typing import Union

from . import opt
from .color_spaces import rgb
from ._private._utils import typeit

__all__ = [
    'VideoCapture',
    'VideoWriter',
    'VideoReader',
    'Video'
]


class VideoInterface:
    """Base class for video operations.
    
    This class provides common functionality for both VideoCapture and VideoWriter classes.
    It implements context manager support and basic video stream operations.
    
    Attributes:
        stream: The underlying OpenCV video stream.
        width (int): Width of the video frames.
        height (int): Height of the video frames.
    """
    stream = None
    width = None
    height = None

    def isOpened(self):
        """Check if the video stream is opened.
        
        Returns:
            bool: True if the stream is opened, False otherwise.
        """
        return self.stream.isOpened()

    @property
    def is_opened(self):
        """bool: Check if the video stream is opened (property version)."""
        return self.isOpened()

    def release(self):
        """Release the video stream and free associated resources.
        
        Note:
            If the stream has not been started, a warning will be issued.
        """
        if self.stream is None:
            warnings.warn("Stream not started")
            return

        self.stream.release()

    @property
    def shape(self):
        """tuple: Shape of the video frames as (width, height)."""
        return self.width, self.height

    def __enter__(self):
        """Enter the runtime context for the video stream.
        
        Returns:
            VideoInterface: This video interface instance.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the runtime context for the video stream.
        
        This method ensures the video stream is released when exiting the context manager.
        """
        self.release()

    close = release
    """Alias for :meth:`release`."""


class VideoCapture(VideoInterface):
    """Video capture class for reading video files or camera streams.
    
    This class provides an enhanced interface for reading video files or capturing
    video from camera devices with automatic color space conversion and error handling.
    
    Attributes:
        stream: The underlying OpenCV VideoCapture stream.
        width (int): Width of the video frames.
        height (int): Height of the video frames.
        frame_cnt (int): Total number of frames in the video.
        fps (int): Frames per second of the video.
    """
    
    def __init__(self, src: Union[Path, str, int]):
        """Initialize a VideoCapture object.

        Args:
            src (Path, str, or int): Source of the video. Can be a file path, directory path,
                or camera index (integer).
                
        Raises:
            IsADirectoryError: If src is a directory.
            FileNotFoundError: If src is a file path that doesn't exist.
            OSError: If the video stream cannot be opened.
            
        Examples:
            >>> # Read from a video file
            >>> cap = VideoCapture('video.mp4')
            >>>
            >>> # Read from a camera (index 0)
            >>> cap = VideoCapture(0)
            >>>
            >>> # Read from a camera (by string index)
            >>> cap = VideoCapture('0')
        """
        if isinstance(src, str) and src.isdecimal():
            src = int(src)
        elif isinstance(src, (str, Path)) and Path(src).is_dir():
            raise IsADirectoryError(str(src))
        elif isinstance(src, Path):
            if not src.is_file():
                raise FileNotFoundError(str(src))
            src = str(src)
        self.stream = BaseVideoCapture(src)
        if not self.is_opened:
            raise OSError("Video from source {} didn't open".format(src))
        self.frame_cnt = round(self.stream.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = round(self.stream.get(cv2.CAP_PROP_FPS))
        self.width = round(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = round(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def now(self):  # current frame
        """int: Current frame number in the video stream.
        
        Raises:
            OSError: If the video is closed.
        """
        if not self.is_opened:
            raise OSError('Video is closed')
        return round(self.stream.get(cv2.CAP_PROP_POS_FRAMES))

    def read(self):
        """Read the next frame from the video stream.
        
        Returns:
            numpy.ndarray: The next video frame.
            
        Raises:
            OSError: If the video is closed.
            StopIteration: If the video has finished.
        """
        if not self.is_opened:
            raise OSError("Video is closed")
        _, frame = self.stream.read()
        if frame is None:
            raise StopIteration('Video has finished')
        if opt.RGB:
            frame = rgb(frame)
        return frame

    def __iter__(self):
        """Return the iterator object (self)."""
        return self

    def __next__(self):
        """Read the next frame from the video stream.
        
        Returns:
            numpy.ndarray: The next video frame.
        """
        frame = self.read()
        return frame

    def rewind(self, nframe):
        """Rewind the video stream to a specific frame.
        
        Args:
            nframe (int): Frame number to rewind to.
            
        Returns:
            VideoCapture: This VideoCapture instance.
        """
        assert isinstance(nframe, int) or (isinstance(nframe, float) and nframe.is_integer())
        assert nframe in range(0, len(self))
        # if 0 <= nframe <= 1:
        self.stream.set(cv2.CAP_PROP_POS_FRAMES, nframe)
        return self

    def __len__(self):
        """int: Total number of frames in the video."""
        if self.frame_cnt < 0:
            return 0
        return self.frame_cnt

    def __getitem__(self, idx):
        """Read a specific frame from the video by index.
        
        Args:
            idx (int): Frame index to read.
            
        Returns:
            numpy.ndarray: The requested video frame.
        """
        self.rewind(idx)
        frame = self.read()
        return frame

    imread = read
    """Alias for :meth:`read`."""
    seek = rewind
    """Alias for :meth:`rewind`."""


class VideoWriter(VideoInterface):
    """Video writer class for creating video files.
    
    This class provides an enhanced interface for writing video files with
    customizable encoding parameters and automatic color space conversion.
    
    Attributes:
        save_path (str): Path to the output video file.
        width (int): Width of the video frames.
        height (int): Height of the video frames.
        fps (int): Frames per second of the output video.
        fourcc: OpenCV FOURCC code for video codec.
        stream: The underlying OpenCV VideoWriter stream.
    """
    
    def __init__(self, save_path, fps=None, fourcc=None, mkdir=False):
        """Initialize a VideoWriter object.

        Args:
            save_path (str or Path): Path to the output video file.
            fps (int, optional): Frames per second. Defaults to opt.FPS.
            fourcc (str or int, optional): FOURCC code for video codec. Defaults to opt.FOURCC.
            mkdir (bool, optional): If True, create parent directories if they don't exist.
                Defaults to False.
                
        Examples:
            >>> # Create a video writer with default settings
            >>> writer = VideoWriter('output.mp4')
            >>>
            >>> # Create a video writer with custom FPS
            >>> writer = VideoWriter('output.mp4', fps=30)
            >>>
            >>> # Create a video writer with string FOURCC
            >>> writer = VideoWriter('output.mp4', fourcc='mp4v')
            >>>
            >>> # Create a video writer with integer FOURCC
            >>> import cv2
            >>> writer = VideoWriter('output.mp4', fourcc=cv2.VideoWriter_fourcc(*'mp4v'))
        """
        if mkdir:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        if isinstance(save_path, Path):
            save_path = str(save_path)
        self.save_path = save_path
        self.width = None
        self.height = None
        self.fps = fps or opt.FPS
        fourcc = fourcc or opt.FOURCC
        if isinstance(fourcc, str):
            fourcc = cv2.VideoWriter_fourcc(*fourcc)
        self.fourcc = fourcc
        self.stream = None

    def isOpened(self):
        """Check if the video writer stream is opened.
        
        Returns:
            bool: True if the stream is opened, False otherwise.
        """
        if self.stream is None:
            return False
        return super().isOpened()

    def write(self, frame: np.ndarray):
        """Write a frame to the video file.
        
        Args:
            frame (numpy.ndarray): Frame to write to the video.
            
        Raises:
            OSError: If the stream is closed.
            AssertionError: If the frame shape doesn't match the video dimensions.
        """
        frame = typeit(frame)
        if self.stream is None:
            self.height, self.width = frame.shape[:2]
            self.stream = BaseVideoWriter(self.save_path, self.fourcc, self.fps, (self.width, self.height))
        if not self.is_opened:
            raise OSError("Stream is closed")
        assert (self.height, self.width) == frame.shape[:2], 'Shape mismatch. Required: {}'.format(self.shape)
        if opt.RGB:
            frame = rgb(frame)
        self.stream.write(frame)

    imwrite = write
    """Alias for :meth:`write`."""


def Video(path, mode='r', **kwds):
    """Factory function to create VideoCapture or VideoWriter instances.
    
    Args:
        path (str or Path): Path to the video file or camera index.
        mode (str, optional): Mode of operation. 'r' for reading, 'w' for writing.
            Defaults to 'r'.
        **kwds: Additional keyword arguments passed to VideoWriter (only in 'w' mode).
        
    Returns:
        VideoCapture or VideoWriter: VideoCapture instance if mode='r',
        VideoWriter instance if mode='w'.
            
    Raises:
        TypeError: If keyword arguments are passed in 'r' mode.
        
    Examples:
        >>> # Create a video reading stream
        >>> reader = Video('video.mp4', mode='r')
        >>>
        >>> # Create a video writing stream
        >>> writer = Video('output.mp4', mode='w')
        >>>
        >>> # Create a video writing stream with custom parameters
        >>> writer = Video('output.mp4', mode='w', fps=30, fourcc='mp4v')
    """
    assert mode in ('r', 'w'), "Mode must be 'r' or 'w'"
    if mode == 'r':
        if kwds:
            raise TypeError(
                "VideoCapture doesn't accept keyword args. If you need VideoWriter then pass mode='w'"
            )
        return VideoCapture(path)
    elif mode == 'w':
        return VideoWriter(path, **kwds)


VideoReader = VideoCapture
"""Alias for :class:`VideoCapture`."""
