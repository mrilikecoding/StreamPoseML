from pose_parser.services.video_data_service import VideoDataService


class ProcessVideoJob:
    @staticmethod
    def process_video(
        input_filename: str,
        video_input_path: str,
        output_data_path: str,
        write_to_file: bool = False,
        configuration: dict = {},
    ):
        """
        This method is intended to wrap the video data service
        which sits in front of the MediaPipe client with a queued job

        Parameters
        -------
            input_filename: str
                the name of the file (webm or mp4 extention right now)
            video_input_path: str
                the location of this video

        Return
        ------

            result: dict
                The dictionary of video data returned by the video processing service


        TODO Here also we should grab some configuration options that can pass
        down into the mediapipe client

        """
        try:
            return VideoDataService().process_video(
                input_filename=input_filename,
                video_input_path=video_input_path,
                output_data_path=output_data_path,
                write_to_file=write_to_file,
                configuration=configuration,
            )
        except:
            raise ProcessVideoJobError(
                f"There was an issue processing video {video_input_path}/{input_filename}"
            )


class ProcessVideoJobError(Exception):
    """Raised when there is an error in the ProcessVideosJob class"""

    pass
