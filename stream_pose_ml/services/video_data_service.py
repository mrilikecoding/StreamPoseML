import time

from stream_pose_ml.utils import path_utility
from stream_pose_ml.blaze_pose.mediapipe_client import MediaPipeClient
from stream_pose_ml.blaze_pose.blaze_pose_sequence import BlazePoseSequence
from stream_pose_ml.serializers.blaze_pose_sequence_serializer import (
    BlazePoseSequenceSerializer,
)


class VideoDataService:
    """
    This class is responsible for providing a service layer to a MediaPipe
    client and passing all the necessary input required to output data
    for each frame
    """

    @staticmethod
    def process_video(
        input_filename: str,
        video_input_path: str,
        write_keypoints_to_file: bool = False,
        output_keypoint_data_path: str | None = None,
        write_serialized_sequence_to_file: bool = False,
        output_sequence_data_path: str | None = None,
        include_geometry: bool = True,
        configuration: dict = {},
        preprocess_video: bool = False,
        id: int | None = None,
        key_off_frame_number: bool = True,
    ) -> dict:
        """Process keypoints from video and use them to model sequence and frame objects.

        The process_video method takes a file name as well as I/O paths,
        spins up a MediaPipeClient to generate raw keypoints,
        then loads them into a BlazePoseSequence object which creates a series of frames
        containing various data

        Args:
            input_filename: str
                The name of the file to process
            video_input_path: str
                The path to the file
            write_keypoints_to_file: bool
                Whether to write the generated keypoints from mediapipe to a file
            write_serialized_sequence_to_file: bool
                Whether to write the serialized sequence data to a file
            output_keypoint_data_path: str
                Where to save the generated keypoint data if written to file
            output_sequence_data_path: str
                Where to save the serialized sequence data if written to file
            include_geometry: bool
                Whether to compute angle and distance measurements in the sequence data
            id: int
                If a specific id should be used for this video
                Will default to a timestamp if not passed into mediapipe client
            configuration: dict
                Configuration options to pass into MediapipeClient
            key_off_frame_number: dict
                When True will key the data dictionary off the frame number - this is handy for extracting specific frames from data

        Returns:
            result: dict
                a serialized BlazePoseSequence dictionary keyed off frame numbers
                This is useful for plucking out specific frames when merging with annotation data

        Raises:
            exception: VideoDataServiceError
        """
        # Set an identifier
        if not id:
            id = int(time.time_ns())
        # Load MPClient and process video
        # TODO parameterize various mediapipe configurables
        # i.e. confidence
        # TODO process multiple times at different confidence?
        # TODO profile this...
        mpc = MediaPipeClient(
            video_input_filename=input_filename,
            video_input_path=video_input_path,
            video_output_prefix=output_keypoint_data_path,
            id=id,
            configuration=configuration,
            preprocess_video=preprocess_video,
        ).process_video()

        if output_keypoint_data_path is None and write_keypoints_to_file:
            raise VideoDataServiceError("No output path specified for keypoint data.")

        if write_keypoints_to_file:
            mpc.write_pose_data_to_file()

        # Compute sequence / frame data
        sequence = BlazePoseSequence(
            name=input_filename,
            sequence=mpc.frame_data_list,
            include_geometry=include_geometry,
        ).generate_blaze_pose_frames_from_sequence()

        # Serialize Data
        sequence_data = BlazePoseSequenceSerializer().serialize(
            sequence, key_off_frame_number=key_off_frame_number
        )

        if output_sequence_data_path is None and write_serialized_sequence_to_file:
            raise VideoDataServiceError(
                "No output path specified for serialzied sequence data."
            )

        if write_serialized_sequence_to_file:
            path_utility.write_to_json_file(
                output_sequence_data_path,
                f"{input_filename}_sequence.json",
                sequence_data,
            )

        return sequence_data


class VideoDataServiceError(Exception):
    """Raise when something is wrong with VideoDataService"""

    pass
