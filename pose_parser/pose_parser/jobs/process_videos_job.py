import os
from pose_parser.jobs.process_video_job import ProcessVideoJob, ProcessVideoJobError


class ProcessVideosJob:
    """
    This class encapsulates processing a batch of videos

    WIP - the idea here will be to use a job queue via Redis but just getting the basic idea here first
    """

    @staticmethod
    def process_videos(
        src_videos_path: str,
        output_data_path: str,
        limit: int = None,
        write_to_file: bool = False,
        configuration: dict = {},
    ):
        """
        This method runs subroutine to process each video in source directory

        Parameters
        -------
            src_video_path: str
                the directory where source videos are found
            limit: int
                max number of videos from this directory to process
        Return
        -----
            results: list
                List of serialized video data dictionaries
        """
        # TODO replace with queued job
        job_count = 0
        try:
            results = []
            for root, dir_names, file_names in os.walk(src_videos_path):
                for f in file_names:
                    if job_count < limit:
                        if f.endswith(("webm", "mp4")):
                            result = ProcessVideoJob.process_video(
                                input_filename=f,
                                video_input_path=src_videos_path,
                                output_data_path=output_data_path,
                                write_to_file=write_to_file,
                                configuration=configuration,
                            )
                            results.append(result)
                            job_count += 1
                    else:
                        break
            return results
        except ProcessVideoJobError as e:
            raise ProcessVideosJobError(f"Error processing videos: {e}")


class ProcessVideosJobError(Exception):
    """Raised when there is an error in the ProcessVideosJob class"""

    pass
