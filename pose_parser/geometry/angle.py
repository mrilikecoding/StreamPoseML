import numpy as np

from pose_parser.geometry.vector import Vector


class Angle:
    """
    This is a data structure representing angles between 2d and 3d vectors

    From:
    https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python/13849249#13849249

    """

    name: str
    vector_1: Vector
    vector_2: Vector
    angle_2d: float
    angle_3d: float
    angle_2d_radians: float
    angle_3d_radians: float
    angle_2d_degrees: float
    angle_3d_degrees: float

    def __init__(self, name: str, vector1: Vector, vector2: Vector) -> None:
        """
        Upon initialization this class creates a named angle object
        where the angle between two vectors is calculated in radians
        and degrees with accessible attributes for introspection
        """
        self.name = name
        self.vector_1 = vector1
        self.vector_2 = vector2
        self.angle_2d = self.angle_between(
            vector1.get_coord_tuple()[:2], vector2.get_coord_tuple()[:2]
        )
        self.angle_3d = self.angle_between(
            vector1.get_coord_tuple()[:3], vector2.get_coord_tuple()[:3]
        )
        self.angle_2d_radians = self.angle_2d  # alias
        self.angle_3d_radians = self.angle_3d  # alias
        self.angle_2d_degrees = np.degrees(self.angle_2d)
        self.angle_3d_degrees = np.degrees(self.angle_3d)
        self.angle_2d_normalized = self.angle_between(
            vector1.get_coord_tuple(normalized=True)[:2],
            vector2.get_coord_tuple(normalized=True)[:2],
        )
        self.angle_3d_normalized = self.angle_between(
            vector1.get_coord_tuple(normalized=True)[:3],
            vector2.get_coord_tuple(normalized=True)[:3],
        )
        self.angle_2d_normalized_radians = self.angle_2d  # alias
        self.angle_3d_normalized_radians = self.angle_3d  # alias
        self.angle_2d_normalized_degrees = np.degrees(self.angle_2d)
        self.angle_3d_normalized_degrees = np.degrees(self.angle_3d)

    @staticmethod
    def unit_vector(vector: tuple) -> float:
        """Returns the unit vector of the vector."""
        return vector / np.linalg.norm(vector)

    @staticmethod
    def angle_between(self, vector_1: tuple, vector_2: tuple) -> float:
        """Returns the angle in radians between vectors 'v1' and 'v2'::

        >>> angle_between((1, 0, 0), (0, 1, 0))
        1.5707963267948966
        >>> angle_between((1, 0, 0), (1, 0, 0))
        0.0
        >>> angle_between((1, 0, 0), (-1, 0, 0))
        3.141592653589793

        """
        v1_u = self.unit_vector(vector_1)
        v2_u = self.unit_vector(vector_2)
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
