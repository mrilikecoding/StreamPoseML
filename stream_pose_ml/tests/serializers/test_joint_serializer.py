import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parents[3]  # Adjust if needed
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from unittest.mock import MagicMock

import pytest

from stream_pose_ml.geometry.joint import Joint
from stream_pose_ml.serializers.joint_serializer import JointSerializer


class TestJointSerializer:
    """Test the JointSerializer class."""

    @pytest.fixture
    def joint(self):
        """Create a mock Joint object for testing."""
        joint = MagicMock(spec=Joint)
        joint.name = "test_joint"
        joint.image_dimensions = {"width": 640, "height": 480}
        joint.x = 0.5
        joint.y = 0.6
        joint.z = 0.7
        joint.x_normalized = 320.0
        joint.y_normalized = 288.0
        joint.z_normalized = 350.0
        return joint

    def test_serialize(self, joint):
        """Test the serialize method."""
        # Given
        serializer = JointSerializer()

        # When
        result = serializer.serialize(joint)

        # Then
        assert result == {
            "type": "Joint",
            "name": "test_joint",
            "image_dimensions": {"width": 640, "height": 480},
            "x": 0.5,
            "y": 0.6,
            "z": 0.7,
            "x_normalized": 320.0,
            "y_normalized": 288.0,
            "z_normalized": 350.0,
        }

    def test_serialize_static_method(self, joint):
        """Test the serialize method as a static method."""
        # When
        result = JointSerializer.serialize(joint)

        # Then
        assert result["name"] == "test_joint"
        assert result["type"] == "Joint"
        assert result["x"] == 0.5
        assert result["y"] == 0.6
        assert result["z"] == 0.7
        assert result["x_normalized"] == 320.0
        assert result["y_normalized"] == 288.0
        assert result["z_normalized"] == 350.0
