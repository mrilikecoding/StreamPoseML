class Joint:
    """
    Coordinates of a joint represented in 2D and 3D, normalized and not normalized
    """

    # the name of this joint
    name: str
    # x, y, z as 0-1
    x: float
    y: float
    z: float
    # x, y, z normalized to the image dimensions
    x_normalized: float
    y_normalized: float
    z_normalized: float
    # the image dimensions for the joint's image {"height": 100, "width": 200}
    image_dimensions: dict

    def __init__(self, name, joint_data: dict) -> None:
        """Init a Joint object with required data.

        Args:
            name: str
                The name of this joint
            joint_data: dict
                Ex.
                {
                    'image_dimensions': { 'height': 500, 'width': 1000 },
                    'x': 0.7057283520698547,
                    'y': 1.333446979522705,
                    'z': 0.5175799131393433,
                    'x_normalized': 1354.998435974121,
                    'y_normalized': 1440.1227378845215,
                    'z_normalized': 993.7534332275391,
                }
        Raises:
            exception: JointError
        """
        try:
            required_keys = [
                "x",
                "y",
                "z",
                "x_normalized",
                "y_normalized",
                "z_normalized",
                "image_dimensions",
            ]
            if not all([key in required_keys for key in joint_data]):
                raise JointError(
                    "The required data is missing from the joint data dictionary."
                )

            self.name = name
            self.image_dimensions = joint_data["image_dimensions"]
            self.x = joint_data["x"]
            self.y = joint_data["y"]
            self.z = joint_data["z"]
            self.x_normalized = joint_data["x_normalized"]
            self.y_normalized = joint_data["y_normalized"]
            self.z_normalized = joint_data["z_normalized"]
        except:
            raise JointError("There was an error instantiating the Joint object")

    def get_coord_tuple(self, normalized=False) -> tuple:
        """Get a tuple representation of the joint.

        Args:
            normalized: bool
                If True, return the vectors's normalized values

        Returns:
            joint: tuple[float, float, float]
                The tuple representation of this joint
        """
        if normalized:
            return self.x_normalized, self.y_normalized, self.z_normalized

        return self.x, self.y, self.z


class JointError(Exception):
    """Raise when there is an error in the Joint class"""

    pass
