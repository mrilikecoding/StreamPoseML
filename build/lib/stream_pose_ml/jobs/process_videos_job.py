from stream_pose_ml.utils import path_utility
from stream_pose_ml.jobs.process_video_job import ProcessVideoJob


class ProcessVideosJob:
    """This class encapsulates processing a batch of videos.

    WIP - the idea here will be to use a job queue via Redis but just getting the basic idea here first
    """

    @staticmethod
    def process_videos(
        src_videos_path: str | None = None,
        output_keypoints_data_path: str | None = None,
        output_sequence_data_path: str | None = None,
        write_keypoints_to_file: bool = False,
        write_serialized_sequence_to_file: bool = False,
        configuration: dict | None = None,
        limit: int | None = None,
        preprocess_video: bool = False,
        return_output: bool = True,
    ) -> list | dict:
        """
        This method runs subroutine to process each video in source directory

        Args:
            src_video_path: str
                the directory where source videos are found
            limit: int
                max number of videos from this directory to process - useful for testing
            write_keypoints_to_file: bool
                whether to write keypoints data to a json file at the output data directory
            write_serialized_sequence_to_file: bool
                whether to write sequence data to a json file at the output data directory
            output_keypoints_data_path: str
                the location to store keypoint data
            output_sequence_data_path: str
                the location to store sequence data
            configuration: dict
                options to pass down to mediapipe client

        Returns:
            results: list
                List of serialized video data dictionaries
                OR
            path_map: dict
                Location of keypoints and sequences



        Raises:
            exception: ProcessVideosJobError
                when no source videos path is present, either passed or in CONFIG
        """
        # TODO replace with queued job
        job_count = 0
        results = []
        video_files = []
        # TODO better way to enforce filetypes?
        for extension in ["webm", "mp4"]:
            video_files += path_utility.get_file_paths_in_directory(
                src_videos_path, extension
            )
        number_to_process = len(video_files)
        for video in video_files:
            if bool(limit) and job_count == limit:
                break
            filename = path_utility.get_file_name(video)
            video_input_path = path_utility.get_base_path(video)
            result = ProcessVideoJob.process_video(
                input_filename=filename,
                video_input_path=video_input_path,
                output_keypoint_data_path=output_keypoints_data_path,
                output_sequence_data_path=output_sequence_data_path,
                write_keypoints_to_file=write_keypoints_to_file,
                write_serialized_sequence_to_file=write_serialized_sequence_to_file,
                configuration=configuration,
                preprocess_video=preprocess_video,
            )
            results.append(result)
            job_count += 1
            print(f"{job_count}/{number_to_process} completed: {filename}.")
        if return_output:
            return results
        else:
            return {
                "keypoints_path": output_keypoints_data_path,
                "sequence_path": output_sequence_data_path,
            }


class ProcessVideosJobError(Exception):
    """Raised when there is an error in the ProcessVideosJob class"""

    pass
