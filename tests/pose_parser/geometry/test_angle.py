import unittest

from stream_pose_ml.geometry.angle import Angle
from stream_pose_ml.geometry.joint import Joint
from stream_pose_ml.geometry.vector import Vector


class TestAngle(unittest.TestCase):
    @classmethod
    def setUpClass(self) -> None:
        self.joint_1 = Joint(
            "joint_1",
            {
                "image_dimensions": {"height": 100, "width": 100},
                "x": 0,
                "y": 0,
                "z": 0,
                "x_normalized": 0,
                "y_normalized": 0,
                "z_normalized": 0,
            },
        )
        self.joint_2 = Joint(
            "joint_2",
            {
                "image_dimensions": {"height": 100, "width": 100},
                "x": 1,
                "y": 1,
                "z": 1,
                "x_normalized": 100,
                "y_normalized": 100,
                "z_normalized": 100,
            },
        )
        self.joint_3 = Joint(
            "joint_3",
            {
                "image_dimensions": {"height": 100, "width": 100},
                "x": 1,
                "y": 0,
                "z": 0,
                "x_normalized": 100,
                "y_normalized": 0,
                "z_normalized": 0,
            },
        )
        self.joint_4 = Joint(
            "joint_4",
            {
                "image_dimensions": {"height": 100, "width": 100},
                "x": 0,
                "y": 1,
                "z": 1,
                "x_normalized": 0,
                "y_normalized": 100,
                "z_normalized": 100,
            },
        )
        self.vector_1 = Vector("v1", joint_1=self.joint_1, joint_2=self.joint_2)
        self.vector_2 = Vector("v2", joint_1=self.joint_3, joint_2=self.joint_4)

    @classmethod
    def tearDownClass(self) -> None:
        # cleanup
        return super().tearDown(self)

    def test_unit_vector(self):
        """
        GIVEN an Angle object
        WHEN unit vector is called on a directional vector
        THEN a unit representation of that vector is returned
        """
        angle = Angle("test", self.vector_1, self.vector_2)
        unit = angle.unit_vector((100, 0, 0))
        self.assertListEqual(list(unit), list([1, 0, 0]))

    def test_angle_between(self):
        """
        GIVEN an Angle object
        WHEN angle between is passed two directional vectors (2D or 3D)
        THEN the angle in radians is returned between the vectors
        """
        angle = Angle("test", self.vector_1, self.vector_2)
        self.assertEqual(angle.angle_between((1, 0, 0), (0, 1, 0)), 1.5707963267948966)
        self.assertEqual(angle.angle_between((1, 0, 0), (1, 0, 0)), 0.0)
        self.assertEqual(angle.angle_between((1, 0, 0), (-1, 0, 0)), 3.141592653589793)

    def test_init(self):
        """
        GIVEN an Angle class
        WHEN an Angle object is instantiated with two Vector objects
        THEN the angle between the two vectors is computed in 2d and 3d
        """
        angle = Angle("test", self.vector_1, self.vector_2)
        self.assertIsInstance(angle, Angle)
        self.assertIsInstance(angle.angle_2d, float)
        self.assertIsInstance(angle.angle_3d, float)
