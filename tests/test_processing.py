import pytest
import numpy as np
import cv3
import cv2

np.random.seed(10)

def test_threshold_defaults():
    """Test threshold function with default parameters."""
    img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    
    # Test with default parameters
    thresh = cv3.threshold(img, 100)
    
    # Compare with OpenCV
    _, expected = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
    
    assert np.array_equal(thresh, expected)


def test_threshold_with_max_value():
    """Test threshold function with custom max value."""
    img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    
    # Test with custom max value
    thresh = cv3.threshold(img, 100, max=128)
    
    # Compare with OpenCV
    _, expected = cv2.threshold(img, 100, 128, cv2.THRESH_BINARY)
    
    assert np.array_equal(thresh, expected)


def test_threshold_with_type_string():
    """Test threshold function with string type parameter."""
    img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    
    # Test with string type
    thresh = cv3.threshold(img, 100, type='binary_inv')
    
    # Compare with OpenCV
    _, expected = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY_INV)
    
    assert np.array_equal(thresh, expected)


def test_threshold_with_type_flag():
    """Test threshold function with flag type parameter."""
    img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    
    # Test with flag type
    thresh = cv3.threshold(img, 100, type=cv2.THRESH_TRUNC)
    
    # Compare with OpenCV
    _, expected = cv2.threshold(img, 100, 255, cv2.THRESH_TRUNC)
    
    assert np.array_equal(thresh, expected)

def test_threshold_relative():
    """Test threshold function with relative threshold value."""
    img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    max_val = img.max()
    
    # Test with relative threshold (0.5 * max_val)
    thresh = cv3.threshold(img, 0.5, rel=True)
    
    # Compare with OpenCV (0.5 * max_val is the relative threshold)
    _, expected = cv2.threshold(img, 0.5 * max_val, 255, cv2.THRESH_BINARY)
    
    assert np.array_equal(thresh, expected)



def test_threshold_all_types():
    """Test all available threshold types."""
    img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    
    types = [
        ('binary', cv2.THRESH_BINARY),
        ('binary_inv', cv2.THRESH_BINARY_INV),
        ('trunc', cv2.THRESH_TRUNC),
        ('tozero', cv2.THRESH_TOZERO),
        ('tozero_inv', cv2.THRESH_TOZERO_INV)
    ]
    
    for type_str, type_flag in types:
        # Test string type
        thresh_str = cv3.threshold(img, 100, type=type_str)
        
        # Test flag type
        thresh_flag = cv3.threshold(img, 100, type=type_flag)
        
        # Compare with OpenCV
        _, expected = cv2.threshold(img, 100, 255, type_flag)
        
        assert np.array_equal(thresh_str, expected)
        assert np.array_equal(thresh_flag, expected)

def test_threshold_opt_type():
    """Test threshold function using opt.THRESHOLD_TYPE."""
    img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    
    # Save original type
    original_type = cv3.opt.THRESHOLD_TYPE
    
    try:
        # Set to a different type
        cv3.opt.THRESHOLD_TYPE = cv2.THRESH_BINARY_INV
        
        # Test with default type (should use opt.THRESHOLD_TYPE)
        thresh = cv3.threshold(img, 100)
        
        # Compare with OpenCV
        _, expected = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY_INV)
        
        assert np.array_equal(thresh, expected)
    finally:
        # Restore original type
        cv3.opt.THRESHOLD_TYPE = original_type



def test_threshold_invalid_type():
    """Test threshold function with invalid type parameter."""
    img = np.zeros((100, 100), dtype=np.uint8)
    
    # Test with invalid string type
    with pytest.raises(AssertionError):
        cv3.threshold(img, 100, type='invalid_type')
    
    # Test with invalid flag type
    with pytest.raises(AssertionError):
        cv3.threshold(img, 100, type=999999)

def test_threshold_invalid_image():
    """Test threshold function with invalid image."""
    # Test with color image (3 channels)
    img_color = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    
    with pytest.raises(AssertionError):
        cv3.threshold(img_color, 100)
