import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parents[3]  # Adjust if needed
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from unittest.mock import MagicMock

import pytest

from stream_pose_ml.geometry.distance import Distance
from stream_pose_ml.geometry.joint import Joint
from stream_pose_ml.geometry.vector import Vector
from stream_pose_ml.serializers.distance_serializer import DistanceSerializer


class TestDistanceSerializer:
    """Test the DistanceSerializer class."""

    @pytest.fixture
    def joint(self):
        """Create a mock Joint object for testing."""
        joint = MagicMock(spec=Joint)
        joint.name = "test_joint"
        return joint

    @pytest.fixture
    def vector(self):
        """Create a mock Vector object for testing."""
        vector = MagicMock(spec=Vector)
        vector.name = "test_vector"
        return vector

    @pytest.fixture
    def distance(self, joint, vector):
        """Create a mock Distance object for testing."""
        distance = MagicMock(spec=Distance)
        distance.name = "test_distance"
        distance.joint = joint
        distance.vector = vector
        distance.distance_2d = 10.5
        distance.distance_3d = 15.7
        distance.distance_2d_normalized = 0.5
        distance.distance_3d_normalized = 0.7
        return distance

    def test_serialize(self, distance):
        """Test the serialize method."""
        # Given
        serializer = DistanceSerializer()

        # When
        result = serializer.serialize(distance)

        # Then
        assert result == {
            "type": "Distance",
            "name": "test_distance",
            "joint_name": "test_joint",
            "vector_name": "test_vector",
            "distance_2d": 10.5,
            "distance_3d": 15.7,
            "distance_2d_normalized": 0.5,
            "distance_3d_normalized": 0.7,
        }

    def test_serialize_static_method(self, distance):
        """Test the serialize method as a static method."""
        # When
        result = DistanceSerializer.serialize(distance)

        # Then
        assert result["name"] == "test_distance"
        assert result["type"] == "Distance"
        assert result["joint_name"] == "test_joint"
        assert result["vector_name"] == "test_vector"
        assert result["distance_2d"] == 10.5
        assert result["distance_3d"] == 15.7
        assert result["distance_2d_normalized"] == 0.5
        assert result["distance_3d_normalized"] == 0.7
