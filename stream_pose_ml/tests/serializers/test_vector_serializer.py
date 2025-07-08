import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Add the project root to the Python path
project_root = Path(__file__).parents[3]  # Adjust if needed
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# ruff: noqa: E402
from stream_pose_ml.geometry.joint import Joint
from stream_pose_ml.geometry.vector import Vector
from stream_pose_ml.serializers.vector_serializer import VectorSerializer


class TestVectorSerializer:
    """Test the VectorSerializer class."""

    @pytest.fixture
    def joint1(self):
        """Create a mock Joint object for testing."""
        joint = MagicMock(spec=Joint)
        joint.name = "joint1"
        return joint

    @pytest.fixture
    def joint2(self):
        """Create a mock Joint object for testing."""
        joint = MagicMock(spec=Joint)
        joint.name = "joint2"
        return joint

    @pytest.fixture
    def vector(self, joint1, joint2):
        """Create a mock Vector object for testing."""
        vector = MagicMock(spec=Vector)
        vector.name = "test_vector"
        vector.joint_1 = joint1
        vector.joint_2 = joint2
        vector.direction_2d = [0.5, 0.5]
        vector.direction_3d = [0.3, 0.3, 0.3]
        vector.direction_reverse_2d = [-0.5, -0.5]
        vector.direction_reverse_3d = [-0.3, -0.3, -0.3]
        vector.x1 = 10
        vector.y1 = 20
        vector.z1 = 30
        vector.x2 = 15
        vector.y2 = 25
        vector.z2 = 35
        vector.x1_normalized = 0.1
        vector.y1_normalized = 0.2
        vector.z1_normalized = 0.3
        vector.x2_normalized = 0.15
        vector.y2_normalized = 0.25
        vector.z2_normalized = 0.35
        return vector

    def test_serialize(self, vector):
        """Test the serialize method."""
        # Given
        serializer = VectorSerializer()

        # When
        result = serializer.serialize(vector)

        # Then
        assert result == {
            "type": "Vector",
            "name": "test_vector",
            "joint_1_name": "joint1",
            "joint_2_name": "joint2",
            "direction_2d": [0.5, 0.5],
            "direction_3d": [0.3, 0.3, 0.3],  # Fixed: Now using the correct 3D value
            "direction_reverse_2d": [-0.5, -0.5],
            "direction_reverse_3d": [
                -0.3,
                -0.3,
                -0.3,
            ],  # Fixed: Now using the correct 3D value
            "x1": 10,
            "y1": 20,
            "z1": 30,
            "x2": 15,
            "y2": 25,
            "z2": 35,
            "x1_normalized": 0.1,
            "y1_normalized": 0.2,
            "z1_normalized": 0.3,
            "x2_normalized": 0.15,
            "y2_normalized": 0.25,
            "z2_normalized": 0.35,
        }

    def test_serialize_static_method(self, vector):
        """Test the serialize method as a static method."""
        # When
        result = VectorSerializer.serialize(vector)

        # Then
        assert result["name"] == "test_vector"
        assert result["type"] == "Vector"
        assert result["joint_1_name"] == "joint1"
        assert result["joint_2_name"] == "joint2"
        assert result["direction_2d"] == [0.5, 0.5]
        assert result["direction_3d"] == [0.3, 0.3, 0.3]
        assert result["x1"] == 10
        assert result["y1"] == 20
        assert result["z1"] == 30
        assert result["x1_normalized"] == 0.1
        assert result["y1_normalized"] == 0.2
        assert result["z1_normalized"] == 0.3
