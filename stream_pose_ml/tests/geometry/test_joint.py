"""Tests for the Joint class."""

import sys
from pathlib import Path

import pytest

from stream_pose_ml.geometry.joint import Joint, JointError

# Add the project root to the Python path
project_root = Path(__file__).parents[
    3
]  # /Users/nathangreen/Development/stream_pose_ml
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


class TestJoint:
    """Tests for the Joint class."""

    @pytest.fixture
    def valid_joint_data(self):
        """Returns valid joint data for testing."""
        return {
            "image_dimensions": {"height": 500, "width": 1000},
            "x": 0.7,
            "y": 0.8,
            "z": 0.9,
            "x_normalized": 700.0,
            "y_normalized": 400.0,
            "z_normalized": 900.0,
        }

    def test_init_with_valid_data(self, valid_joint_data):
        """
        GIVEN valid joint data
        WHEN a Joint is initialized
        THEN the Joint is created with the correct attributes
        """
        # Act
        joint = Joint(name="test_joint", joint_data=valid_joint_data)

        # Assert
        assert joint.name == "test_joint"
        assert joint.image_dimensions == valid_joint_data["image_dimensions"]
        assert joint.x == valid_joint_data["x"]
        assert joint.y == valid_joint_data["y"]
        assert joint.z == valid_joint_data["z"]
        assert joint.x_normalized == valid_joint_data["x_normalized"]
        assert joint.y_normalized == valid_joint_data["y_normalized"]
        assert joint.z_normalized == valid_joint_data["z_normalized"]

    def test_init_with_missing_required_key(self, valid_joint_data):
        """
        GIVEN joint data missing a required key
        WHEN a Joint is initialized
        THEN JointError is raised
        """
        # Arrange
        # Remove a required key
        del valid_joint_data["x"]

        # Act & Assert
        with pytest.raises(JointError):
            Joint(name="test_joint", joint_data=valid_joint_data)

    def test_get_coord_tuple_not_normalized(self, valid_joint_data):
        """
        GIVEN a valid Joint
        WHEN get_coord_tuple is called with normalized=False
        THEN a tuple of non-normalized coordinates is returned
        """
        # Arrange
        joint = Joint(name="test_joint", joint_data=valid_joint_data)

        # Act
        result = joint.get_coord_tuple(normalized=False)

        # Assert
        assert isinstance(result, tuple)
        assert len(result) == 3
        assert result == (
            valid_joint_data["x"],
            valid_joint_data["y"],
            valid_joint_data["z"],
        )

    def test_get_coord_tuple_normalized(self, valid_joint_data):
        """
        GIVEN a valid Joint
        WHEN get_coord_tuple is called with normalized=True
        THEN a tuple of normalized coordinates is returned
        """
        # Arrange
        joint = Joint(name="test_joint", joint_data=valid_joint_data)

        # Act
        result = joint.get_coord_tuple(normalized=True)

        # Assert
        assert isinstance(result, tuple)
        assert len(result) == 3
        assert result == (
            valid_joint_data["x_normalized"],
            valid_joint_data["y_normalized"],
            valid_joint_data["z_normalized"],
        )
