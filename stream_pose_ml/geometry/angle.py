import numpy as np

from stream_pose_ml.geometry.vector import Vector


class Angle:
    """This is a data structure representing angles between 2d and 3d vectors.

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
        """Initialize an Angle object.

        Upon initialization this class creates a named angle object
        where the angle between two vectors is calculated in radians
        and degrees with accessible attributes for introspection
        """
        try:
            self.name = name
            self.vector_1 = vector1
            self.vector_2 = vector2
            self.angle_2d = self.angle_between(
                vector1.direction_2d, vector2.direction_2d
            )
            self.angle_3d = self.angle_between(
                vector1.direction_3d, vector2.direction_3d
            )
            self.angle_2d_radians = self.angle_2d  # alias
            self.angle_3d_radians = self.angle_3d  # alias
            self.angle_2d_degrees = np.degrees(self.angle_2d)
            self.angle_3d_degrees = np.degrees(self.angle_3d)
        except:
            raise AngleError("There was a problem instantiating the angle.")

    def unit_vector(self, vector: tuple) -> float:
        """Determine the directional unit vector of a passed tuple of two points.

        Given a passed vector with beginning and end points
        Obtain the directional vector then compute the unit vector

        Args:
            vector: tuple[float, float, float] | tuple[float, float]
                3D or 2D directional vector - e.g. Vector(...).direction_2d
                Note - this is not two points but rather the computed vector direction from the points

        Returns:
            unit_vector: tuple[float, float, float] | tuple[float, float]
                A unit vector in 3D or 2D - note, this will be the same
                regardless of normalized values
        """
        try:
            return vector / np.linalg.norm(vector)
        except:
            raise AngleError("There was an error computing the unit vector")

    def angle_between(self, vector_1: tuple, vector_2: tuple) -> float:
        """Given two directional vectors (i.e. Vector(..).direction_3d) get the angle.

        Returns:
            the angle in radians between vectors 'v1' and 'v2'

        e.g.
        angle_between((1, 0, 0), (0, 1, 0))
        1.5707963267948966
        angle_between((1, 0, 0), (1, 0, 0))
        0.0
        angle_between((1, 0, 0), (-1, 0, 0))
        3.141592653589793

        Args:
            vector_1: tuple[float, float, float]
            vector_2: tuple[float, float, float]

        Returns:
            angle: float
                Angle in radians

        Raises:
            exception: AngleError
                When there's an issue computing the angle

        """
        try:
            v1_u = self.unit_vector(vector_1)
            v2_u = self.unit_vector(vector_2)
            return np.arccos(np.clip(np.dot(v1_u, v2_u.T), -1.0, 1.0))
        except:
            raise AngleError("There was an error computing the vector angle.")


class AngleError(Exception):
    """Raises: when there is an error in the Angle class"""
