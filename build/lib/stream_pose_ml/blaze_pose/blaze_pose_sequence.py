from stream_pose_ml.blaze_pose.blaze_pose_frame import BlazePoseFrame
from stream_pose_ml.blaze_pose.enumerations import BlazePoseJoints


class BlazePoseSequence:
    """A sequence of BlazePoseFrames.

    It validates they have the right shape and then creates a BlazePoseFrame for each pass frame

    """

    sequence_data: list[dict]  # a list of frame data dicts for keypoints / metadata
    frames: list[
        BlazePoseFrame
    ]  # a list of BlazePoseFrames representing the sequence data
    joint_positions: list[str]  # required keys for a non-empty joint position object
    include_geometry: bool  # compute angles / distance measure for each frame based on joint data

    def __init__(
        self, name: str, sequence: list = [], include_geometry: bool = False
    ) -> None:
        self.name = name
        self.joint_positions = [joint.name for joint in BlazePoseJoints]
        for frame in sequence:
            if not self.validate_pose_schema(frame_data=frame):
                raise BlazePoseSequenceError("Validation error!")

        self.sequence_data = sequence
        self.include_geometry = include_geometry
        self.frames = []

    def validate_pose_schema(self, frame_data: dict):
        """This method is responsible for ensuring data meets the required schema.

        Args:

            frame_data: dict
                a MediaPipeClient.frame_data_list entry conforming to proper schema

        Returns:
            valid: bool
                returns True if the data is valid

        Raises:
            exception: BlazePoseSequenceError
                Raises: an exception if there is a problem with validation
        """
        required_keys = [
            "sequence_id",
            "sequence_source",
            "frame_number",
            "image_dimensions",
            "joint_positions",
        ]
        # verify required top level keys are present
        for key in required_keys:
            if key not in frame_data:
                raise BlazePoseSequenceError(
                    f"Validation error - {key} is missing from frame data"
                )

        joint_positions = frame_data["joint_positions"]

        # it is possible there is no joint position data for a frame
        if not joint_positions:
            return True

        # if there is joint position data, ensure all keys are present
        for pos in self.joint_positions:
            if pos not in joint_positions:
                raise BlazePoseSequenceError(
                    f"Validation error - {pos} is missing from joint position data"
                )

        return True

    def generate_blaze_pose_frames_from_sequence(self) -> "BlazePoseSequence":
        """Create frame objects from this object's sequence of data.

        For each frame data in the sequence data list
        generate a BlazePoseFrame object and add it to the list of frames

        Returns:
            self: BlazePoseSequence
                returns this instance for chaining to init
        """
        try:
            for frame_data in self.sequence_data:
                bpf = BlazePoseFrame(
                    frame_data=frame_data,
                    generate_angles=self.include_geometry,
                    generate_distances=self.include_geometry,
                )
                self.frames.append(bpf)
            return self
        except:
            raise BlazePoseSequenceError(
                "There was a problem generating a BlazePoseFrame"
            )

    def serialize_sequence_data(self):
        """This method returns a list of serialized frame data.

        Returns:
            frames_json: list[dict]

        Raises:
            exception: BlazePoseSequenceError
                raises BlazePoseSequenceError if there's a problem
        """
        try:
            return [frame.serialize_frame_data() for frame in self.frames]
        except:
            raise BlazePoseSequenceError("Error serializing frames")


class BlazePoseSequenceError(Exception):
    """
    Raises: when there is an error in the BlazePoseSequence class
    """

    pass
