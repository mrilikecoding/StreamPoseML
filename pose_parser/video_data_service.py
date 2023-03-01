import time

from pose_parser.blaze_pose.mediapipe_client import MediaPipeClient
from pose_parser.blaze_pose.blaze_pose_sequence import BlazePoseSequence
from pose_parser.serializers.blaze_pose_sequence_serializer import (
    BlazePoseSequenceSerializer,
)


class VideoDataService:
    def __init__(self) -> None:
        pass

    @staticmethod
    def process_video(
        input_filename: str,
        video_input_path: str,
        output_data_path: str,
        include_geometry: bool = True,
        id: int = None,
    ) -> dict:
        """
        The process_video method takes a file name as well as I/O paths,
        spins up a MediaPipeClient to generate raw keypoints,
        then loads them into a BlazePoseSequence object which creates a series of frames
        containing various data
        """
        # Set an identifier
        if not id:
            id = int(time.time_ns())
        # Load MPClient and process video
        mpc = MediaPipeClient(
            video_input_filename=input_filename,
            video_input_path=video_input_path,
            video_output_prefix=output_data_path,
            id=id,
        ).process_video()

        # Compute sequence / frame data
        sequence = BlazePoseSequence(
            name=input_filename,
            sequence=mpc.frame_data_list,
            include_geometry=include_geometry,
        ).generate_blaze_pose_frames_from_sequence()

        # Serialize Data
        data = BlazePoseSequenceSerializer().serialize(sequence, key_off_frame_number=True)

        return data
