from stream_pose_ml.services.video_data_service import VideoDataService


class ProcessVideoJob:
    """This class is a wrapper for the VideoDataService.

    This is meant to be run by a job queue.
    """

    @staticmethod
    def process_video(
        input_filename: str,
        video_input_path: str,
        output_keypoint_data_path: str,
        output_sequence_data_path: str,
        write_keypoints_to_file: bool = False,
        write_serialized_sequence_to_file: bool = False,
        configuration: dict = {},
        preprocess_video: bool = False,
    ) -> dict:
        """This method is intended to wrap the video data service which sits in front of the MediaPipe client with a queued job

        Args:
            input_filename: str
                the name of the file (webm or mp4 extention right now)
            video_input_path: str
                the location of this video
            output_data_path: str
                where to put keypoint_data

        Returns:
            result: dict
                The dictionary of video data returned by the video processing service

        Raises:
            ProcessVideoJob error when file writes are wanted but not paths are provided


        """
        if write_keypoints_to_file and output_keypoint_data_path is None:
            raise ProcessVideoJobError(
                "No output location specified for keypoints files."
            )
        if write_serialized_sequence_to_file and output_sequence_data_path is None:
            raise ProcessVideoJobError(
                "No output location specified for sequence data files."
            )

        return VideoDataService().process_video(
            input_filename=input_filename,
            video_input_path=video_input_path,
            write_keypoints_to_file=write_keypoints_to_file,
            output_keypoint_data_path=output_keypoint_data_path,
            write_serialized_sequence_to_file=write_serialized_sequence_to_file,
            output_sequence_data_path=output_sequence_data_path,
            configuration=configuration,
            preprocess_video=preprocess_video,
        )


class ProcessVideoJobError(Exception):
    """Raised when there is an error in the ProcessVideosJob class"""

    pass
