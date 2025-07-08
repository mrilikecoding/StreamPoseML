import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Add the project root to the Python path
project_root = Path(__file__).parents[3]  # Adjust if needed
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# ruff: noqa: E402
from stream_pose_ml.geometry.angle import Angle
from stream_pose_ml.geometry.vector import Vector
from stream_pose_ml.serializers.angle_serializer import AngleSerializer


class TestAngleSerializer:
    """Test the AngleSerializer class."""

    @pytest.fixture
    def vector1(self):
        """Create a mock Vector object for testing."""
        vector = MagicMock(spec=Vector)
        vector.name = "vector1"
        return vector

    @pytest.fixture
    def vector2(self):
        """Create a mock Vector object for testing."""
        vector = MagicMock(spec=Vector)
        vector.name = "vector2"
        return vector

    @pytest.fixture
    def angle(self, vector1, vector2):
        """Create a mock Angle object for testing."""
        angle = MagicMock(spec=Angle)
        angle.name = "test_angle"
        angle.vector_1 = vector1
        angle.vector_2 = vector2
        angle.angle_2d = 0.5  # ~28.6 degrees
        angle.angle_2d_degrees = 28.6
        angle.angle_3d = 0.7  # ~40.1 degrees
        angle.angle_3d_degrees = 40.1
        return angle

    def test_serialize(self, angle):
        """Test the serialize method."""
        # Given
        serializer = AngleSerializer()

        # When
        result = serializer.serialize(angle)

        # Then
        assert result == {
            "type": "Angle",
            "vector_1": "vector1",
            "vector_2": "vector2",
            "name": "test_angle",
            "angle_2d": 0.5,
            "angle_2d_degrees": 28.6,
            "angle_3d": 0.7,
            "angle_3d_degrees": 40.1,
        }

    def test_serialize_static_method(self, angle):
        """Test the serialize method as a static method."""
        # When
        result = AngleSerializer.serialize(angle)

        # Then
        assert result["name"] == "test_angle"
        assert result["type"] == "Angle"
        assert result["vector_1"] == "vector1"
        assert result["vector_2"] == "vector2"
        assert result["angle_2d"] == 0.5
        assert result["angle_2d_degrees"] == 28.6
        assert result["angle_3d"] == 0.7
        assert result["angle_3d_degrees"] == 40.1
