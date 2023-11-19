import unittest

from stream_pose_ml.geometry.vector import Vector
from stream_pose_ml.geometry.joint import Joint
from stream_pose_ml.geometry.distance import Distance


class TestDistance(unittest.TestCase):
    def test_init(self):
        """
        GIVEN a Distance class
        WHEN a distance is instantiated with a Vector and a Joint
        THEN a new Distance object is created with all the expected attributes
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
        j3 = Joint(
            name="j3",
            joint_data={
                "image_dimensions": {"height": 100, "width": 100},
                "x": 1,
                "y": 0,
                "z": 1,
                "x_normalized": 100,
                "y_normalized": 0,
                "z_normalized": 100,
            },
        )

        vector = Vector("test", joint_1=j1, joint_2=j2)
        distance = Distance("test_dist", joint=j3, vector=vector)
        self.assertIsInstance(distance, Distance)
        self.assertEqual(distance.distance_2d, 0.5)
        self.assertEqual(distance.distance_2d_normalized, 50)

    def test_distance_from_joint_to_vector_midpoint(self):
        """
        GIVEN a Distance object
        WHEN distance from joint to vector midpoint is called
        THEN a float value is returned representing euclidean distance from the joint coords to the vector midpoint
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
        j3 = Joint(
            name="j3",
            joint_data={
                "image_dimensions": {"height": 100, "width": 100},
                "x": 1,
                "y": 0,
                "z": 1,
                "x_normalized": 100,
                "y_normalized": 0,
                "z_normalized": 100,
            },
        )

        vector = Vector("test", joint_1=j1, joint_2=j2)
        distance = Distance("test_dist", joint=j3, vector=vector)
        # 3d
        dist_to_mid = distance.distance_from_joint_to_vector_midpoint(
            joint_coords=j3.get_coord_tuple(), vector=vector.get_coord_tuple()
        )
        self.assertEqual(0.71, round(dist_to_mid, 2))
        # 3d normalized
        dist_to_mid = distance.distance_from_joint_to_vector_midpoint(
            joint_coords=j3.get_coord_tuple(normalized=True),
            vector=vector.get_coord_tuple(normalized=True),
        )
        self.assertEqual(71, round(dist_to_mid, 0))
        # 2d
        vector_2d = (vector.get_coord_tuple()[0][:2], vector.get_coord_tuple()[1][:2])
        dist_to_mid = distance.distance_from_joint_to_vector_midpoint(
            joint_coords=j3.get_coord_tuple()[:2], vector=vector_2d
        )
        self.assertEqual(0.50, round(dist_to_mid, 2))
        # 2d normalized
        vector_2d = (
            vector.get_coord_tuple(normalized=True)[0][:2],
            vector.get_coord_tuple(normalized=True)[1][:2],
        )
        dist_to_mid = distance.distance_from_joint_to_vector_midpoint(
            joint_coords=j3.get_coord_tuple(normalized=True)[:2], vector=vector_2d
        )
        self.assertEqual(50, round(dist_to_mid, 2))
