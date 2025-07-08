"""Tests for the Vector class."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from stream_pose_ml.geometry.joint import Joint
from stream_pose_ml.geometry.vector import Vector, VectorError

# Add the project root to the Python path
project_root = Path(__file__).parents[
    3
]  # /Users/nathangreen/Development/stream_pose_ml
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


class TestVector:
    """Tests for the Vector class."""

    @pytest.fixture
    def mock_joint1(self):
        """Returns a mock Joint for testing."""
        joint = MagicMock(spec=Joint)
        joint.name = "joint1"
        joint.x = 1.0
        joint.y = 2.0
        joint.z = 3.0
        joint.x_normalized = 100.0
        joint.y_normalized = 200.0
        joint.z_normalized = 300.0
        joint.get_coord_tuple.side_effect = lambda normalized=False: (
            (100.0, 200.0, 300.0) if normalized else (1.0, 2.0, 3.0)
        )
        return joint

    @pytest.fixture
    def mock_joint2(self):
        """Returns another mock Joint for testing."""
        joint = MagicMock(spec=Joint)
        joint.name = "joint2"
        joint.x = 4.0
        joint.y = 5.0
        joint.z = 6.0
        joint.x_normalized = 400.0
        joint.y_normalized = 500.0
        joint.z_normalized = 600.0
        joint.get_coord_tuple.side_effect = lambda normalized=False: (
            (400.0, 500.0, 600.0) if normalized else (4.0, 5.0, 6.0)
        )
        return joint

    def test_init(self, mock_joint1, mock_joint2):
        """
        GIVEN two Joint objects
        WHEN a Vector is initialized
        THEN the Vector is created with the correct attributes
        """
        # Act
        vector = Vector(name="test_vector", joint_1=mock_joint1, joint_2=mock_joint2)

        # Assert
        assert vector.name == "test_vector"
        assert vector.joint_1 is mock_joint1
        assert vector.joint_2 is mock_joint2

        # Check coordinates
        assert vector.x1 == mock_joint1.x
        assert vector.y1 == mock_joint1.y
        assert vector.z1 == mock_joint1.z
        assert vector.x1_normalized == mock_joint1.x_normalized
        assert vector.y1_normalized == mock_joint1.y_normalized
        assert vector.z1_normalized == mock_joint1.z_normalized

        assert vector.x2 == mock_joint2.x
        assert vector.y2 == mock_joint2.y
        assert vector.z2 == mock_joint2.z
        assert vector.x2_normalized == mock_joint2.x_normalized
        assert vector.y2_normalized == mock_joint2.y_normalized
        assert vector.z2_normalized == mock_joint2.z_normalized

        # Check direction vectors
        assert vector.direction_2d == (vector.x2 - vector.x1, vector.y2 - vector.y1)
        assert vector.direction_3d == (
            vector.x2 - vector.x1,
            vector.y2 - vector.y1,
            vector.z2 - vector.z1,
        )
        assert vector.direction_reverse_2d == (
            vector.x1 - vector.x2,
            vector.y1 - vector.y2,
        )
        assert vector.direction_reverse_3d == (
            vector.x1 - vector.x2,
            vector.y1 - vector.y2,
            vector.z1 - vector.z2,
        )

    def test_init_error(self, mock_joint1):
        """
        GIVEN invalid arguments
        WHEN a Vector is initialized
        THEN VectorError is raised
        """
        # Act & Assert
        with pytest.raises(VectorError, match="Error instantiating Vector object"):
            # Passing a non-Joint object should cause an error
            Vector(name="test_vector", joint_1=mock_joint1, joint_2="not_a_joint")

    def test_get_coord_tuple_not_normalized(self, mock_joint1, mock_joint2):
        """
        GIVEN a valid Vector
        WHEN get_coord_tuple is called with normalized=False
        THEN a tuple of tuples with non-normalized coordinates is returned
        """
        # Arrange
        vector = Vector(name="test_vector", joint_1=mock_joint1, joint_2=mock_joint2)

        # Act
        result = vector.get_coord_tuple(normalized=False)

        # Assert
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result[0] == mock_joint1.get_coord_tuple(normalized=False)
        assert result[1] == mock_joint2.get_coord_tuple(normalized=False)
        assert mock_joint1.get_coord_tuple.called_with(normalized=False)
        assert mock_joint2.get_coord_tuple.called_with(normalized=False)

    def test_get_coord_tuple_normalized(self, mock_joint1, mock_joint2):
        """
        GIVEN a valid Vector
        WHEN get_coord_tuple is called with normalized=True
        THEN a tuple of tuples with normalized coordinates is returned
        """
        # Arrange
        vector = Vector(name="test_vector", joint_1=mock_joint1, joint_2=mock_joint2)

        # Act
        result = vector.get_coord_tuple(normalized=True)

        # Assert
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result[0] == mock_joint1.get_coord_tuple(normalized=True)
        assert result[1] == mock_joint2.get_coord_tuple(normalized=True)
        assert mock_joint1.get_coord_tuple.called_with(normalized=True)
        assert mock_joint2.get_coord_tuple.called_with(normalized=True)

    def test_get_coord_tuple_error(self, mock_joint1, mock_joint2):
        """
        GIVEN a Vector where get_coord_tuple will raise an error
        WHEN get_coord_tuple is called
        THEN VectorError is raised
        """
        # Arrange
        vector = Vector(name="test_vector", joint_1=mock_joint1, joint_2=mock_joint2)
        mock_joint1.get_coord_tuple.side_effect = Exception("Test error")

        # Act & Assert
        with pytest.raises(VectorError, match="Error obtaining tuple coordinates"):
            vector.get_coord_tuple()
