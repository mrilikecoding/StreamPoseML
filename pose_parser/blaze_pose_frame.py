from pose_parser.geometry.joint import Joint
from pose_parser.geometry.vector import Vector
from pose_parser.geometry.angle import Angle
from pose_parser.geometry.distance import Distance


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

    def validate_joint_position_data(self, joint_positions: dict) -> bool:
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

    def generate_angle_measurements(self) -> None:
        """
        If we have joint positions create necessary baseline joints
        Then define baseline vectors
        Then create a series of angle measurements based on the current pose frame

        """
        if not self.has_joint_positions:
            raise BlazePoseFrameError(
                f"There are no joint data to generate angles from"
            )

        # TODO call these for each key by making Distance and Angles
        # distances = self.open_pose_angle_definition_map()
        # angles = self.open_pose_angle_definition_map()

    def open_pose_distance_definition_map(self):
        """
        This method is responsible for translating Blaze pose joints into
        OpenPose domain joints / vectors and returning a map to vectors
        that can be used to generate distance calculations between a joint
        and vectors

        Return
        -----
            distance_definition_map: dict
                A map from named joints to vectors for use in
                distance calculation
        """
        # create a translation map from openpose distance spec to joint->vector distance
        return {
            "distance_point_0__line_25_26": ("nose", "plumb_line"),
            "distance_point_1__line_25_26": ("neck", "plumb_line"),
            "distance_point_2__line_25_26": ("right_shoulder", "plumb_line"),
            "distance_point_3__line_25_26": ("right_elbow", "plumb_line"),
            "distance_point_4__line_25_26": ("right_wrist", "plumb_line"),
            "distance_point_5__line_25_26": ("left_shoulder", "plumb_line"),
            "distance_point_6__line_25_26": ("left_elbow", "plumb_line"),
            "distance_point_7__line_25_26": ("left_wrist", "plumb_line"),
            "distance_point_8__line_25_26": ("mid_hip", "plumb_line"),
            "distance_point_9__line_25_26": ("right_hip", "plumb_line"),
            "distance_point_10__line_25_26": ("right_knee", "plumb_line"),
            "distance_point_11__line_25_26": ("right_ankle", "plumb_line"),
            "distance_point_12__line_25_26": ("left_hip", "plumb_line"),
            "distance_point_13__line_25_26": ("left_knee", "plumb_line"),
            "distance_point_14__line_25_26": ("left_ankle", "plumb_line"),
            "distance_point_15__line_25_26": ("right_eye", "plumb_line"),
            "distance_point_16__line_25_26": ("left_eye", "plumb_line"),
            "distance_point_17__line_25_26": ("right_ear", "plumb_line"),
            "distance_point_18__line_25_26": ("left_ear", "plumb_line"),
            "distance_point_19__line_25_26": ("left_foot_index", "plumb_line"),
            # "distance_point_20__line_25_26": ("", "plumb_line"), # skipping as there's not a good analog
            "distance_point_21__line_25_26": ("left_heel", "plumb_line"),
            "distance_point_22__line_25_26": ("right_foot_index", "plumb_line"),
            # "distance_point_23__line_25_26": ("", "plumb_line"), #skipping as there's not a good analog
            "distance_point_24__line_25_26": ("right_heel", "plumb_line"),
        }

    def open_pose_angle_definition_map(self):
        """
        This method is responsible for translating Blaze pose joints into
        OpenPose domain joints / vectors and returning a map to vectors
        that can be used to generate angle calculations between vectors

        Return
        -----
            angle_definition_map: dict
                A map from named angles to created vectors for use in
                angle generation
        """
        self.joints["neck"] = self.get_average_joint(
            name="neck", joint_1="left_shoulder", joint_2="right_shoulder"
        )
        self.joints["mid_hip"] = self.get_average_joint(
            name="mid_hip", joint_1="left_hip", joint_2="right_hip"
        )

        # create necessary vectors translating from openpose domain
        # i.e. create plumb line vector as the angle basis - joints must
        # exist in self.joints

        # OpenPose 25_26
        self.vectors["plumb_line"] = self.get_vector("plumb_line", "neck", "mid_hip")
        # OpenPose 0_1
        self.vectors["nose_neck"] = self.get_vector("nose_neck", "nose", "neck")
        # OpenPose 1_8
        self.vectors["neck_mid_hip"] = self.get_vector(
            "neck_mid_hip", "neck", "mid_hip"
        )
        # OpenPose 1_2
        self.vectors["neck_right_shoulder"] = self.get_vector(
            "neck_right_shoulder", "neck", "right_shoulder"
        )
        # OpenPose 1_5
        self.vectors["neck_left_shoulder"] = self.get_vector(
            "neck_left_shoulder", "neck", "left_shoulder"
        )
        # OpenPose 2_3
        self.vectors["right_shoulder_right_elbow"] = self.get_vector(
            "right_shoulder_right_elbow", "right_shoulder", "right_elbow"
        )
        # OpenPose 3_4
        self.vectors["right_elbow_right_wrist"] = self.get_vector(
            "right_elbow_right_wrist", "right_elbow", "right_wrist"
        )
        # OpenPose 5_6
        self.vectors["left_shoulder_left_elbow"] = self.get_vector(
            "left_shoulder_left_elbow", "left_shoulder", "left_elbow"
        )
        # OpenPose 6_7
        self.vectors["right_elbow_right_wrist"] = self.get_vector(
            "left_elbow_left_wrist", "left_elbow", "left_wrist"
        )
        # OpenPose 8_9
        self.vectors["mid_hip_right_hip"] = self.get_vector(
            "mid_hip_right_hip", "mid_hip", "right_hip"
        )
        # OpenPose 9_10
        self.vectors["right_hip_right_knee"] = self.get_vector(
            "right_hip_right_knee", "right_hip", "right_knee"
        )
        # OpenPose 10_11
        self.vectors["right_knee_right_ankle"] = self.get_vector(
            "right_knee_right_ankle", "right_knee", "right_ankle"
        )
        # OpenPose 8_12
        self.vectors["mid_hip_left_hip"] = self.get_vector(
            "mid_hip_left_hip", "mid_hip", "left_hip"
        )
        # OpenPose 12_13
        self.vectors["left_hip_left_knee"] = self.get_vector(
            "left_hip_left_knee", "left_hip", "left_knee"
        )
        # OpenPose 13_14
        self.vectors["left_knee_left_ankle"] = self.get_vector(
            "left_knee_left_ankle", "left_knee", "left_ankle"
        )
        # OpenPose 0_15
        self.vectors["nose_right_eye"] = self.get_vector(
            "nose_right_eye", "nose", "right_eye"
        )
        # OpenPose 15_17
        self.vectors["right_eye_right_ear"] = self.get_vector(
            "right_eye_right_ear", "right_eye", "right_ear"
        )
        # OpenPose 0_16
        self.vectors["nose_left_eye"] = self.get_vector(
            "nose_left_eye", "nose", "left_eye"
        )
        # OpenPose 16_18
        self.vectors["left_eye_left_ear"] = self.get_vector(
            "left_eye_left_ear", "left_eye", "left_ear"
        )
        # OpenPose 14_19 -- note OpenPose is "Left Big Toe" - here using left foot index from Blaze
        self.vectors["left_ankle_left_foot_index"] = self.get_vector(
            "left_ankle_left_foot_index", "left_ankle", "left_foot_index"
        )
        # OpenPose 19_20 -- Skipping - not a good analog for this
        # OpenPose 14_21
        self.vectors["left_ankle_left_heel"] = self.get_vector(
            "left_ankle_left_heel", "left_ankle", "left_heel"
        )
        # OpenPose 11_22 -- note OpenPose is "Right Big Toe" - here using right foot index from Blaze
        self.vectors["right_ankle_right_foot_index"] = self.get_vector(
            "right_ankle_right_foot_index", "right_ankle", "right_foot_index"
        )
        # OpenPose 22_23 -- Skipping - not a good analog for this
        # OpenPose 11_24
        self.vectors["right_ankle_right_heel"] = self.get_vector(
            "right_ankle_right_heel", "right_ankle", "right_heel"
        )

        # create a translation map from openpose angle spec to vector->vector angle
        return {
            "line_0_1__line_25_26": ("", "plumb_line"),
            "line_1_8__line_25_26": ("", "plumb_line"),
            "line_1_2__line_25_26": ("", "plumb_line"),
            "line_1_5__line_25_26": ("", "plumb_line"),
            "line_1_5__line_25_26": ("", "plumb_line"),
            "line_2_3__line_25_26": ("", "plumb_line"),
            "line_3_4__line_25_26": ("", "plumb_line"),
            "line_5_6__line_25_26": ("", "plumb_line"),
            "line_6_7__line_25_26": ("", "plumb_line"),
            "line_8_9__line_25_26": ("", "plumb_line"),
            "line_9_10__line_25_26": ("", "plumb_line"),
            "line_10_11__line_25_26": ("", "plumb_line"),
            "line_8_12__line_25_26": ("", "plumb_line"),
            "line_12_13__line_25_26": ("", "plumb_line"),
            "line_13_14__line_25_26": ("", "plumb_line"),
            "line_0_15__line_25_26": ("", "plumb_line"),
            "line_15_17__line_25_26": ("", "plumb_line"),
            "line_0_16__line_25_26": ("", "plumb_line"),
            "line_16_18__line_25_26": ("", "plumb_line"),
            "line_14_19__line_25_26": ("", "plumb_line"),
            "line_19_20__line_25_26": ("", "plumb_line"),
            "line_14_21__line_25_26": ("", "plumb_line"),
            "line_11_22__line_25_26": ("", "plumb_line"),
            "line_22_23__line_25_26": ("", "plumb_line"),
            "line_11_24__line_25_26": ("", "plumb_line"),
            "line_0_1__line_1_8": ("", "plumb_line"),
            "line_0_1__line_1_2": ("", "plumb_line"),
            "line_1_8__line_8_12": ("", "plumb_line"),
            "line_1_2__line_2_3": ("", "plumb_line"),
            "line_1_5__line_5_6": ("", "plumb_line"),
            "line_2_3__line_3_4": ("", "plumb_line"),
            "line_5_6__line_6_7": ("", "plumb_line"),
            "line_8_9__line_9_10": ("", "plumb_line"),
            "line_10_11__line_11_24": ("", "plumb_line"),
            "line_8_12__line_12_13": ("", "plumb_line"),
            "line_12_13__line_13_14": ("", "plumb_line"),
            "line_13_14__line_14_19": ("", "plumb_line"),
            "line_0_15__line_15_17": ("", "plumb_line"),
            "line_0_16__line_16_18": ("", "plumb_line"),
            "line_0_16__line_16_18": ("", "plumb_line"),
            "line_14_19__line_19_20": ("", "plumb_line"),
            "line_14_21__line_11_22": ("", "plumb_line"),
            "line_11_22__line_22_23": ("", "plumb_line"),
        }

    def get_vector(self, name: str, joint_name_1: str, joint_name_2: str) -> Vector:
        """
        This method creates and returns a new vector object based on existing joints

        Parameters
        ---------
            name: str
                The name of this vector for reference
            joint_name_1: str
                The name of the first joint - this should be an existing key in self.joints
            joint_name_2: str
                The name of the second joint - this should be an existing key in self.joints

        Return
        -------
            vector: Vector
                a Vector object representing 2d and 3d vector between passed joints
        """
        return Vector(name, self.joints[joint_name_1], self.joints[joint_name_2])

    def get_average_joint(self, name: str, joint_1: str, joint_2: str) -> Joint:
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
