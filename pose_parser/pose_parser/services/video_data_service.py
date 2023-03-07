import time

from pose_parser.blaze_pose.mediapipe_client import (
    MediaPipeClient,
    MediaPipeClientError,
)
from pose_parser.blaze_pose.blaze_pose_sequence import (
    BlazePoseSequence,
    BlazePoseSequenceError,
)
from pose_parser.serializers.blaze_pose_sequence_serializer import (
    BlazePoseSequenceSerializer,
)


class VideoDataService:
    """
    This class is responsible for providing a service layer to a MediaPipe
    client and passing all the necessary input required to output data
    for each frame
    """

    def __init__(self) -> None:
        pass

    @staticmethod
    def process_video(
        input_filename: str,
        video_input_path: str,
        output_data_path: str,
        include_geometry: bool = True,
        id: int = None,
        write_to_file: bool = False,
        configuration={},
        key_off_frame_number: bool = True,
    ) -> dict:
        """
        The process_video method takes a file name as well as I/O paths,
        spins up a MediaPipeClient to generate raw keypoints,
        then loads them into a BlazePoseSequence object which creates a series of frames
        containing various data

        Paramters
        -------
            input_filename: str
                The name of the file to process
            video_input_path: str
                The path to the file
            write_to_file: bool
                Whether to write the generated data to a file
            output_data_path: str
                Where to save the generated data if written to file
            include_geometry: bool
                Whether to compute angle and distance measurements
            id: int
                If a specific id should be used for this video
                Will default to a timestamp if not passed into mediapipe client
            configuration: dict
                Configuration options to pass into MediapipeClient
            key_off_frame_number: dict
                When True will key the data dictionary off the frame number - this is handy for extracting specific frames from data

        Return
        ----
            result: dict
                a serialized BlazePoseSequence dictionary keyed off frame numbers
                This is useful for plucking out specific frames when merging with annotation data

        Raise:
            exception: BlazePoseSequenceError
                Raised if there's an issue with generating a sequence object
            exception: MediaPipeClientError
                Raised if there's an issue with the MediaPipe Client
        """
        try:
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
                video_output_prefix=output_data_path,
                id=id,
                configuration=configuration,
            ).process_video()

            if write_to_file:
                mpc.write_pose_data_to_file()

            # Compute sequence / frame data
            sequence = BlazePoseSequence(
                name=input_filename,
                sequence=mpc.frame_data_list,
                include_geometry=include_geometry,
            ).generate_blaze_pose_frames_from_sequence()

            # Serialize Data
            data = BlazePoseSequenceSerializer().serialize(
                sequence, key_off_frame_number=key_off_frame_number
            )

            return data
        except MediaPipeClientError as e:
            raise VideoDataServiceError(
                f"There was an issue running the MediaPipe Client: {e}"
            )
        except BlazePoseSequenceError as e:
            raise VideoDataServiceError(
                f"There was an issue computing the BlazePoseSequence: {e}"
            )


class VideoDataServiceError(Exception):
    """Raise when something is wrong with VideoDataService"""

    pass
