from stream_pose_ml.geometry.joint import Joint


class Vector:
    """This is a data structure representing a vector of two joints."""

    name: str  # name of this vector
    joint_1: Joint
    joint_2: Joint
    x1: float
    y1: float
    z1: float
    x1_normalized: float
    y1_normalized: float
    z1_normalized: float
    x2: float
    y2: float
    z2: float
    x2_normalized: float
    y2_normalized: float
    z2_normalized: float
    direction_2d: tuple[float, float]
    direction_3d: tuple[float, float, float]
    direction_reverse_2d: tuple[float, float]
    direction_reverse_3d: tuple[float, float, float]

    def __init__(self, name: str, joint_1: Joint, joint_2: Joint) -> None:
        """Initialize a Vector object.

        Upon intitialization, build a representation of a vector using the two
        passed joints for both normalized and original values

        Args:
            name: str
                the name of this Vector
            joint_1: Joint
                the first joint in the vector
            joint_2: Joint
                the second joint in the vector
        Raises:
            exception: VectorError

        """
        try:
            self.name = name
            self.joint_1 = joint_1
            self.joint_2 = joint_2
            self.x1 = joint_1.x
            self.y1 = joint_1.y
            self.z1 = joint_1.z
            self.x1_normalized = joint_1.x_normalized
            self.y1_normalized = joint_1.y_normalized
            self.z1_normalized = joint_1.z_normalized

            self.x2 = joint_2.x
            self.y2 = joint_2.y
            self.z2 = joint_2.z
            self.x2_normalized = joint_2.x_normalized
            self.y2_normalized = joint_2.y_normalized
            self.z2_normalized = joint_2.z_normalized
            self.direction_2d = (self.x2 - self.x1, self.y2 - self.y1)
            self.direction_3d = (
                self.x2 - self.x1,
                self.y2 - self.y1,
                self.z2 - self.z1,
            )
            self.direction_reverse_2d = (self.x1 - self.x2, self.y1 - self.y2)
            self.direction_reverse_3d = (
                self.x1 - self.x2,
                self.y1 - self.y2,
                self.z1 - self.z2,
            )
        except:
            raise VectorError("There was an issue instantiating the Vector object")

    def get_coord_tuple(self, normalized: bool = False):
        """This method returns a tuple representation of the vector points.

        Args:
            normalized: bool
                If True, return the vectors's normalized values

        Returns:
            joint: tuple[tuple[float, float, float], tuple[float, float, float]]
                The tuple representation of this vector
        """
        try:
            if normalized:
                return (
                    self.joint_1.get_coord_tuple(normalized=True),
                    self.joint_2.get_coord_tuple(normalized=True),
                )

            return (
                self.joint_1.get_coord_tuple(normalized=False),
                self.joint_2.get_coord_tuple(normalized=False),
            )
        except:
            raise VectorError("There was an error obtaining the tuple coordinates")


class VectorError(Exception):
    """Raised when there is an error in the Vector class"""

    pass
