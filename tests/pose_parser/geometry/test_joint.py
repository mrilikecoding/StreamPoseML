import unittest

from stream_pose_ml.geometry.joint import Joint


class TestJoint(unittest.TestCase):
    def test_init(self):
        """
        GIVEN a Joint class
        WHEN the right data is passed in to instantiate a Joint object
        THEN a Joint object is created with all the expected attributes
        """
        data = {
            "image_dimensions": {"height": 500, "width": 1000},
            "x": 0.7057283520698547,
            "y": 1.333446979522705,
            "z": 0.5175799131393433,
            "x_normalized": 1354.998435974121,
            "y_normalized": 1440.1227378845215,
            "z_normalized": 993.7534332275391,
        }
        joint = Joint("test", joint_data=data)
        self.assertEqual("test", joint.name)
        self.assertEqual(joint.image_dimensions, data["image_dimensions"])
        self.assertEqual(joint.y, data["y"])
        self.assertEqual(joint.x, data["x"])
        self.assertEqual(joint.y, data["y"])
        self.assertEqual(joint.z, data["z"])
        self.assertEqual(joint.x_normalized, data["x_normalized"])
        self.assertEqual(joint.y_normalized, data["y_normalized"])
        self.assertEqual(joint.z_normalized, data["z_normalized"])

    def test_get_coord_tuple(self):
        """
        GIVEN a joint object
        WHEN get_coord_tuple is called with or without normalization flag
        THEN a tuple representation of the joint coordinates is returned
        """
        data = {
            "image_dimensions": {"height": 500, "width": 1000},
            "x": 0.7057283520698547,
            "y": 1.333446979522705,
            "z": 0.5175799131393433,
            "x_normalized": 1354.998435974121,
            "y_normalized": 1440.1227378845215,
            "z_normalized": 993.7534332275391,
        }
        joint = Joint("test", joint_data=data)
        x, y, z = joint.get_coord_tuple()
        x_n, y_n, z_n = joint.get_coord_tuple(normalized=True)
        self.assertEqual(x, data["x"])
        self.assertEqual(y, data["y"])
        self.assertEqual(z, data["z"])
        self.assertEqual(x_n, data["x_normalized"])
        self.assertEqual(y_n, data["y_normalized"])
        self.assertEqual(z_n, data["z_normalized"])
