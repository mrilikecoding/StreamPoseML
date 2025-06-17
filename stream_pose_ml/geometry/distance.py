import numpy as np

from stream_pose_ml.geometry.vector import Vector
from stream_pose_ml.geometry.joint import Joint


class Distance:
    """2D and 3D representations of distance between a joint and midpoint of vector."""

    name: str  # name of this distance
    joint: Joint
    vector: Vector
    distance_2d: float
    distance_3d: float
    distance_2d_normalized: float
    distance_3d_normalized: float

    def __init__(self, name: str, joint: Joint, vector: Vector) -> None:
        """Initialized a Distance object.

        Upon init, compute the 2D and 3D distance from the Joint's coordinates
        to the midpoint of the passed Vector object

        Args:
            name: str
                The name of this distance measure
            joint: Joint
                A joint object to compute the distance to the vector with
            vector: Vector
                A vector object to compute the midpoint's distance to joint with

        Raises:
            exception: DistanceError

        """
        self.name = name
        self.joint = joint
        self.vector = vector

        vector_tuple = self.vector.get_coord_tuple()
        vector_tuple_normalized = self.vector.get_coord_tuple(normalized=True)
        self.distance_2d = self.distance_from_joint_to_vector_midpoint(
            self.joint.get_coord_tuple()[:2],
            (vector_tuple[0][:2], vector_tuple[1][:2]),
        )
        self.distance_3d = self.distance_from_joint_to_vector_midpoint(
            self.joint.get_coord_tuple()[:3],
            (vector_tuple[0][:3], vector_tuple[1][:3]),
        )
        self.distance_2d_normalized = self.distance_from_joint_to_vector_midpoint(
            self.joint.get_coord_tuple(normalized=True)[:2],
            (vector_tuple_normalized[0][:2], vector_tuple_normalized[1][:2]),
        )
        self.distance_3d_normalized = self.distance_from_joint_to_vector_midpoint(
            self.joint.get_coord_tuple(normalized=True)[:3],
            (vector_tuple_normalized[0][:3], vector_tuple_normalized[1][:3]),
        )

    def distance_from_joint_to_vector_midpoint(
        self, joint_coords: tuple, vector: tuple
    ):
        """Get distance from joint to a vector's midpoint.

        This method determines the midpoint of the passed vector tuple
        and then returns the Euclidean distance (L2 Norm) between
        the joint coords and the midpoint.

        Args:
            joint_coords: tuple[float]
                (x, y, z) coordintaes for a joint
            vector: tuple[tuple[float, float], tuplep[float, float]]
                ((x, y, z), (x, y, z)) for a vector

        Returns:
            distance: float
                The Euclidean distance (L2 Norm) between the joint and the midpoint of the vector.
                Also see:
                https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html

        """
        if len(joint_coords) == 3:
            x1, y1, z1 = vector[0]
            x2, y2, z2 = vector[1]
            midpoint = ((x1 + x2 / 2), (y1 + y2 / 2), (z1 + z2) / 2)
        else:
            x1, y1 = vector[0]
            x2, y2 = vector[1]
            midpoint = ((x1 + x2 / 2), (y1 + y2 / 2))

        dist = np.linalg.norm(np.array(joint_coords) - np.array(midpoint))
        return dist


class DistanceError(Exception):
    """Raised when there is an error in the Distance class"""

    pass
