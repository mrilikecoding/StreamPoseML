from pose_parser.blaze_pose_frame import BlazePoseFrame


class BlazePoseSequence:
    """
    This class represents a sequence of BlazePoseFrames

    It validates they have the right shape and then creates a BlazePoseFrame for each pass frame

    """

    sequence_data: list  # a list of frame data dicts for keypoints / metadata
    frames: list  # a list of BlazePoseFrames representing the sequence data
    joint_positions: list  # required keys for a non-empty joint position object

    def __init__(self, sequence: list = []) -> None:
        self.joint_positions = [
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
        for frame in sequence:
            if not self.validate_pose_schema(frame_data=frame):
                raise BlazePoseSequenceError("Validation error!")

        self.sequence_data = sequence
        self.frames = []

    def validate_pose_schema(self, frame_data: dict):
        """
        This method is responsible for ensuring data meets the required schema

        Parameters
        ------

            frame_data: dict
                a MediaPipeClient.frame_data_list entry conforming to proper schema

        Returns
        -------
            valid: bool
                returns True if the data is valid

        Raises
        _____
            exception: BlazePoseSequenceError
                Raises an exception if there is a problem with validation
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

    def generate_blaze_pose_frames_from_sequence(self):
        for frame_data in self.sequence_data:
            bpf = BlazePoseFrame(frame_data=frame_data)
            self.frames.append(bpf)


class BlazePoseSequenceError(Exception):
    """
    Raise when there is an error in the BlazePoseSequence class
    """

    pass
