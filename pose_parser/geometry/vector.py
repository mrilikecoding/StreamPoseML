from pose_parser.geometry.joint import Joint


class Vector:
    """This is a data structure representing a vector of two joints"""

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

    def __init__(self, name: str, joint_1: Joint, joint_2: Joint) -> None:
        """
        Upon intitialization, build a representation of a vector using the two
        passed joints for both normalized and original values
        """
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

    def get_coord_tuple(self, normalized: bool = False):
        """
        This method returns a tuple representation of the vector

        Parameters
        --------
            normalized: bool
                If True, return the vectors's normalized values

        Return
        ------
            joint: tuple[tuple[float, float, float], tuple[float, float, float]]
                The tuple representation of this vector
        """
        if normalized:
            return (
                self.joint_1.get_coord_tuple(normalized=True),
                self.joint_2.get_coord_tuple(normalized=True),
            )

        return (
            self.joint_1.get_coord_tuple(normalized=False),
            self.joint_2.get_coord_tuple(normalized=False),
        )


class VectorError(Exception):
    """Raised when there is an error in the Vector class"""

    pass
