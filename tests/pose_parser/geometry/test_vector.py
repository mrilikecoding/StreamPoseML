import unittest

from stream_pose_ml.geometry.joint import Joint
from stream_pose_ml.geometry.vector import Vector


class TestVector(unittest.TestCase):
    def test_init(self):
        """
        GIVEN a Vector class
        WHEN a vector is instantiated with two Joints
        THEN a new Vector object is created with all the expected attributes
        """
        j1 = Joint(
            name="j1",
            joint_data={
                "image_dimensions": {"height": 100, "width": 100},
                "x": 1,
                "y": 0,
                "z": 0,
                "x_normalized": 100,
                "y_normalized": 0,
                "z_normalized": 0,
            },
        )
        j2 = Joint(
            name="j2",
            joint_data={
                "image_dimensions": {"height": 100, "width": 100},
                "x": 0,
                "y": 1,
                "z": 1,
                "x_normalized": 0,
                "y_normalized": 100,
                "z_normalized": 100,
            },
        )

        vector = Vector("test", joint_1=j1, joint_2=j2)
        self.assertTupleEqual(vector.joint_1.get_coord_tuple(), (j1.x, j1.y, j1.z))
        self.assertTupleEqual(vector.joint_2.get_coord_tuple(), (j2.x, j2.y, j2.z))

    def test_get_coord_tuple(self):
        """
        GIVEN a vector object
        WHEN get_coord_tuple is called with or without normalization flag
        THEN a tuple representation of the vector coordinates is returned
        """
        j1 = Joint(
            name="j1",
            joint_data={
                "image_dimensions": {"height": 100, "width": 100},
                "x": 1,
                "y": 0,
                "z": 0,
                "x_normalized": 100,
                "y_normalized": 0,
                "z_normalized": 0,
            },
        )
        j2 = Joint(
            name="j2",
            joint_data={
                "image_dimensions": {"height": 100, "width": 100},
                "x": 0,
                "y": 1,
                "z": 1,
                "x_normalized": 0,
                "y_normalized": 1,
                "z_normalized": 1,
            },
        )

        vector = Vector("test", joint_1=j1, joint_2=j2)
        coords = vector.get_coord_tuple()
        self.assertTupleEqual(coords[0], (j1.x, j1.y, j1.z))
        self.assertTupleEqual(coords[1], (j2.x, j2.y, j2.z))
