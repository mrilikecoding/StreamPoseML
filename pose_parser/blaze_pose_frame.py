from pose_parser.geometry.joint import Joint
from pose_parser.geometry.vector import Vector


class BlazePoseFrame:
    """
    This class represents a single frame of BlazePose joint positions
    It stores meta-data related to the frame and also computes angle measurements if joint positions are present
    """

    joint_positions: list
    frame_number: int
    has_joint_positions: bool
    image_dimensions: tuple
    sequence_id: int
    sequence_source: str
    joints: dict
    angles: dict
    vectors: dict

    def __init__(self, frame_data: dict) -> None:
        """
        Initialize this class - passed a dictionary of frame data

        Parameters
        -----
            frame_data: dict
                This is passed from a BlazePoseSequence.sequence_data entry
                ex.
                {
                    'sequence_id': 1677107027968938000,
                    'sequence_source': 'mediapipe',
                    'frame_number': 43,
                    'image_dimensions': {'height': 1080, 'width': 1920},
                    'joint_positions': {
                        'nose': {'x': 12., 'y': 42., 'z': 32., x_normalized: ...}
                        ...
                    }
                }

        """
        self.joint_position_names = [
            "nose",
            "left_eye_inner",
            "left_eye",
            "left_eye_outer",
            "right_eye_inner",
            "right_eye",
            "right_eye_outer",
            "left_ear",
            "right_ear",
            "mouth_left",
            "mouth_right",
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
            "left_pinky",
            "right_pinky",
            "left_index",
            "right_index",
            "left_thumb",
            "right_thumb",
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_anle",
            "left_heel",
            "right_heel",
            "left_foot_index",
            "right_foot_index",
        ]
        self.joints = {}
        self.vectors = {}
        self.angles = {}
        self.frame_number = frame_data["frame_number"]
        self.has_joint_positions = bool(frame_data["joint_positions"])
        self.image_dimensions = frame_data["image_dimensions"]
        self.sequence_id = frame_data["sequence_id"]
        self.sequence_source = frame_data["sequence_source"]
        # if we have joint positions, validate them
        # and instantiate joint objects into dictionary
        if self.has_joint_positions:
            self.validate_joint_position_data(frame_data["joint_positions"])
            self.joint_positions_raw = frame_data["joint_positions"]
            self.joints = self.set_joint_positions()
            self.angles = self.generate_angle_measurements()

    def set_joint_positions(self) -> dict:
        """
        This method takes the raw joint data from every named joint
        and formats a data object to create a Joint object instance

        Returns

            joint_positions: dict
                A joint position dictionary where each key is the name of
                a joint and the value is a dictionary containing position
                data for that joint in this frame instance
        """
        if not self.has_joint_positions:
            raise BlazePoseFrameError("There are no joint positions to set")
        joint_positions = {}
        for joint in self.joint_position_names:
            joints_raw = self.joint_positions_raw
            joint_data = {
                "image_dimensions": self.image_dimensions,
                "x": joints_raw[joint]["x"],
                "y": joints_raw[joint]["y"],
                "z": joints_raw[joint]["z"],
                "x_normalized": joints_raw[joint]["x_normalized"],
                "y_normalized": joints_raw[joint]["y_normalized"],
                "z_normalized": joints_raw[joint]["z_normalized"],
            }
            joint_positions[joint] = Joint(name=joint, joint_data=joint_data)
        return joint_positions

    def validate_joint_position_data(self, joint_positions: dict):
        """
        This method validates that the required keys are present in
        the joint position data

        Parameters
        --------
            joint_positions: dict
                a dictionary of joint position data to be validated

        Returns
        ______
            success: bool
                If all keys are present return true

        Raise
        -----
            BlazePoseFrameError if we are missing a key


        """
        required_joint_keys = [
            "x",
            "y",
            "z",
            "x_normalized",
            "y_normalized",
            "z_normalized",
        ]

        for joint in self.joint_position_names:
            if joint in joint_positions:
                for key in required_joint_keys:
                    if key in joint_positions[joint]:
                        continue
                    else:
                        raise BlazePoseFrameError(
                            f"{key} missing from {joint} position data"
                        )
            else:
                raise BlazePoseFrameError(f"{joint} missing from joint positions dict")

        return True

    def generate_angle_measurements(self):
        if not self.has_joint_positions:
            raise BlazePoseFrameError(
                f"There are no joint data to generate angles from"
            )

        # compute plumb line vector as the angle basis
        self.joints["neck"] = self.get_average_joint(
            name="neck", joint_1="left_shoulder", joint_2="right_shoulder"
        )
        self.joints["mid_hip"] = self.get_average_joint(
            name="mid_hip", joint_1="left_hip", joint_2="right_hip"
        )
        self.vectors["plumb_line"] = self.get_vector("plumb_line", "neck", "mid_hip")

    def get_vector(self, name: str, joint_name_1: str, joint_name_2: str):
        return Vector(name, self.joints[joint_name_1], self.joints[joint_name_2])

    def get_average_joint(self, name: str, joint_1: str, joint_2: str):
        """
        Blaze pose has a specific set of joints. However, some instances we want
        to compute the value of a mid point. In these cases take the average of two
        named joints

        Parameters
        ---------
            name: str
                what to name the joint that is generated
            joint_1: str
                the name of the joint to lookup in self.joints
            joint_2: str
                the name of the joint to lookup in self.joints

        Returns
        -------
            joint: Joint
                a Joint object representing a joint at the average of the passed joints
        """
        x = (self.joints[joint_1].x + self.joints[joint_2].x) / 2
        y = (self.joints[joint_1].y + self.joints[joint_2].y) / 2
        z = (self.joints[joint_1].z + self.joints[joint_2].z) / 2
        x_normalized = (
            self.joints[joint_1].x_normalized + self.joints[joint_2].x_normalized
        ) / 2
        y_normalized = (
            self.joints[joint_1].y_normalized + self.joints[joint_2].y_normalized
        ) / 2
        z_normalized = (
            self.joints[joint_1].z_normalized + self.joints[joint_2].z_normalized
        ) / 2

        return Joint(
            name=name,
            joint_data={
                "x": x,
                "y": y,
                "z": z,
                "x_normalized": x_normalized,
                "y_normalized": y_normalized,
                "z_normalized": z_normalized,
                "image_dimensions": self.image_dimensions,
            },
        )

    def serialize_frame_data(self):
        pass


class BlazePoseFrameError(Exception):
    """
    Raise when there is an error in the BlazePoseFrame class
    """

    pass
