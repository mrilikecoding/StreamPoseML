from stream_pose_ml.geometry.joint import Joint
from stream_pose_ml.geometry.vector import Vector
from stream_pose_ml.geometry.angle import Angle
from stream_pose_ml.geometry.distance import Distance
from stream_pose_ml.blaze_pose.openpose_mediapipe_transformer import (
    OpenPoseMediapipeTransformer,
)
from stream_pose_ml.blaze_pose.enumerations import BlazePoseJoints


class BlazePoseFrame:
    """A single video frame's BlazePose joint positions and various frame data.

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
    distances: dict

    def __init__(
        self,
        frame_data: dict,
        generate_angles: bool = False,
        generate_distances: bool = False,
    ) -> None:
        """
        Initialize frame object.

        Args:
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

            generate_angles: bool
                Default - False. Create angle measurements based on pose data. These are based
                on OpenPose body 25 angle measurements. This will result in the creation of some
                specialized joints and vectors for translating between BlazePose model and OpenPose

            generate_distances: bool
                Default - False. Create distance measurements between joints and vectors based on pose data.
                These are based on OpenPose body 25 distance measurements. This will result in the creation of some
                specialized joints and vectors for translating between BlazePose model and OpenPose

        """
        self.joint_position_names = [joint.name for joint in BlazePoseJoints]
        self.joints = {}
        self.vectors = {}
        self.angles = {}
        self.distances = {}

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

        # By default, we don't have new joints / vectors calculated
        # using openpose specifications...
        self.has_openpose_joints_and_vectors = False
        # So here, if desired, generate angles and distance measures based on
        # OpenPose Body 25 angle / distance measures
        if self.has_joint_positions and (generate_angles or generate_distances):
            self.has_openpose_joints_and_vectors = (
                OpenPoseMediapipeTransformer.create_openpose_joints_and_vectors(self)
            )
            if self.has_openpose_joints_and_vectors and generate_angles:
                angle_map = (
                    OpenPoseMediapipeTransformer.open_pose_angle_definition_map()
                )
                self.angles = self.generate_angle_measurements(angle_map)
            if self.has_openpose_joints_and_vectors and generate_distances:
                distance_map = (
                    OpenPoseMediapipeTransformer.open_pose_distance_definition_map()
                )
                self.distances = self.generate_distance_measurements(distance_map)

    def set_joint_positions(self) -> dict:
        """Take raw joint data and create Joint object.

        This method takes the raw joint data from every named joint
        and formats a data object to create a Joint object instance

        Returns:

            joint_positions: dict
                A joint position dictionary where each key is the name of
                a joint and the value is a dictionary containing position
                data for that joint in this frame instance
        """
        try:
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
        except:
            raise BlazePoseFrameError(
                "There was an error setting the joint positions for the BlazePoseFrame"
            )

    def validate_joint_position_data(self, joint_positions: dict) -> bool:
        """Make sure right data is present in joint position.

        This method validates that the required keys are present in
        the joint position data

        Args:
            joint_positions: dict
                a dictionary of joint position data to be validated

        Returns:
            success: bool
                If all keys are present return true

        Raises:
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

    def generate_distance_measurements(self, distance_map: dict) -> dict:
        """Create distance measurements based on the passed map.

        Args:
            map: dict[str, tuple[str, str]]
                map of openpose definitions to their named joint -> vector distance calculation.
                This is essentially defining the

        Returns:
            distances: dict[str, Distance]
                dictionary mapping the named distance measure to a Distance object



        """
        if not self.has_joint_positions:
            raise BlazePoseFrameError(
                f"There are no joint data to generate distances from"
            )
        if not self.has_openpose_joints_and_vectors:
            self.has_openpose_joints_and_vectors = (
                OpenPoseMediapipeTransformer.create_openpose_joints_and_vectors(self)
            )

        distances = {}
        for name, measure in distance_map.items():
            joint, vector = measure
            distances[name] = Distance(name, self.joints[joint], self.vectors[vector])

        return distances

    def generate_angle_measurements(self, angle_map: dict) -> dict:
        """Create angle measurements based on the passed map.

        Args:
            map: dict[str, tuple[str, str]]
                Keys structured "angle_name": ("vector_name_1", "vector_name_2":)

        Returns:
            angles: dict[str, Angle]
                dictionary mapping the named angle measure to an Angle object
        """
        if not self.has_joint_positions:
            raise BlazePoseFrameError(
                f"There are no joint data to generate angles from"
            )
        if not self.has_openpose_joints_and_vectors:
            self.has_openpose_joints_and_vectors = (
                OpenPoseMediapipeTransformer.create_openpose_joints_and_vectors(self)
            )

        angles = {}

        for name, measure in angle_map.items():
            vector_1, vector_2 = measure
            angles[name] = Angle(name, self.vectors[vector_1], self.vectors[vector_2])

        return angles

    def get_vector(self, name: str, joint_name_1: str, joint_name_2: str) -> Vector:
        """This method creates and returns a new vector object based on existing joints.

        Args:
            name: str
                The name of this vector for reference
            joint_name_1: str
                The name of the first joint - this should be an existing key in self.joints
            joint_name_2: str
                The name of the second joint - this should be an existing key in self.joints

        Returns:
            vector: Vector
                a Vector object representing 2d and 3d vector between passed joints
        """
        return Vector(name, self.joints[joint_name_1], self.joints[joint_name_2])

    def get_average_joint(self, name: str, joint_1: str, joint_2: str) -> Joint:
        """Compute a new joint at coordinate average of two other joints.

        Blaze pose has a specific set of joints. However, some instances we want
        to compute the value of a mid point. In these cases take the average of two
        named joints

        Args:
            name: str
                what to name the joint that is generated
            joint_1: str
                the name of the joint to lookup in self.joints
            joint_2: str
                the name of the joint to lookup in self.joints

        Returns:
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


class BlazePoseFrameError(Exception):
    """
    Raised when there is an error in the BlazePoseFrame class
    """

    pass
