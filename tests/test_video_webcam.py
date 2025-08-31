"""Tests for webcam video capture in cv3.

This module contains tests for webcam video capture functionality provided by the cv3 library.
These tests are specifically for testing video capture from webcam devices (device index 0).

Note: These tests require a webcam to be connected to the system to run successfully.
"""

import cv3

def test_capture_open_webcam():
    with cv3.Video(0):
        pass

    with cv3.Video('0'):
        pass