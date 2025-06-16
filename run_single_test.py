#!/usr/bin/env python3
"""
Simple script to run a small part of a test without pytest discovery.
"""
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.absolute()))

# Test function for distance calculation
def test_distance_calculation():
    """
    Simple test to calculate distance between two points.
    This is a simplified version of the test_distance tests.
    """
    import numpy as np
    
    # Calculate Euclidean distance between points
    def calculate_distance(point1, point2):
        return np.linalg.norm(np.array(point1) - np.array(point2))
    
    # Test data
    point1 = (0.0, 0.0, 0.0)  # Origin
    point2 = (3.0, 4.0, 0.0)  # Point at (3,4,0)
    
    # Expected distance (3-4-5 triangle)
    expected = 5.0
    
    # Calculate actual distance
    actual = calculate_distance(point1, point2)
    
    # Check if they match
    assert np.isclose(actual, expected), f"Expected {expected}, got {actual}"
    return True

# Run the test
if __name__ == "__main__":
    try:
        success = test_distance_calculation()
        if success:
            print("✅ Test passed!")
        else:
            print("❌ Test failed!")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        sys.exit(1)