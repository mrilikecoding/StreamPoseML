import glob
import os
import json
import time
from pathlib import Path
from pose_parser.utils import path_utility

from pose_parser.services.dataloop_annotation_transformer_service import (
    DataloopAnnotationTransformerService,
)
from pose_parser.services.video_data_service import VideoDataService


class VideoDataDataloopMergeService:
    """Merge Dataloop annotations with video data.

    This class is responsible for searching an annotations directory to find
    relevant annotations for a specified video
    """

    annotations_data_directory: str
    video_directory: str
    sequence_data_directory: str
    process_videos: bool
    output_data_path: str
    output_keypoints_path: str
    annotation_video_map: dict
    video_annotation_map: dict
    merged_data: dict

    def __init__(
        self,
        annotations_data_directory: str,
        output_data_path: str,
        output_keypoints_path: str | None = None,
        sequence_data_directory: str | None = None,
        process_videos: bool = False,
        video_directory: str = None,
    ) -> None:
        """Upon initialization set data source directories and initialize storage dicts.

        This class then creates an annotation map between the source annotations directory
        and the video directory or sequence data directory.

        Args:
            annotations_directory: str
                where do the source annotations live?
            sequence_data_directory: str
                if sequence data has been generated from video, where does it live?
            video_directory: str
                where do the source videos live?
            process_videos: bool
                generate sequence data from videos to use
            output_data_path: str
                where to save the merged data
            output_keypoints_path: str
                where to put keypoint data
        """
        self.annotations_data_directory = annotations_data_directory
        self.video_directory = video_directory
        self.sequence_data_directory = sequence_data_directory
        self.output_data_path = output_data_path
        self.output_keypoints_path = (output_keypoints_path,)
        self.annotation_video_map = {}
        self.video_annotation_map = {}
        self.annotation_sequence_map = {}
        self.sequence_annotation_map = {}
        self.process_videos = process_videos
        self.merged_data = {}

        self.transformer = DataloopAnnotationTransformerService()

        self.create_video_annotation_map()

    def create_video_annotation_map(self):
        """This method reads from the annotations and videos directory to
        create a file path map between videos and annotations.

        Returns:
            success: bool
                Returns: True if operation completes successfully

        Raises:
            exception: VideoDataDataloopMergeServiceError
                Raises:d if there's an error creating this map
        """
        if self.process_videos and self.video_directory is None:
            raise VideoDataDataloopMergeServiceError(
                "There is no source video directory specified to generate video data from."
            )

        annotation_files = path_utility.get_file_paths_in_directory(
            directory=self.annotations_data_directory, extension="json"
        )

        # TODO maybe makes sense to set this in config?
        if self.video_directory:
            valid_extensions = ["webm", "mp4"]
            video_files = path_utility.get_file_paths_in_directory(
                directory=self.video_directory, extension=valid_extensions
            )

        sequence_files = path_utility.get_file_paths_in_directory(
            directory=self.sequence_data_directory, extension="json"
        )

        # create a map between source videos, annotations, and sequence_data
        for annotation_path in annotation_files:
            # load dataloop file and grab source filename
            filename = path_utility.get_file_name(annotation_path, omit_extension=True)

            if self.video_directory:
                for video_path in video_files:
                    if filename in video_path:
                        self.annotation_video_map[annotation_path] = video_path
                        self.video_annotation_map[video_path] = annotation_path

            if self.sequence_data_directory:
                for sequence_path in sequence_files:
                    if filename in sequence_path:
                        self.annotation_sequence_map[annotation_path] = sequence_path
                        self.sequence_annotation_map[sequence_path] = annotation_path

        return True

    def generate_dataset(self, limit: int = None, write_to_file: bool = False) -> dict:
        """Use this object's generated map to create a nested dataset

        This method is reponsible for calling the video data service
        based on the map and merging the returned data into a dataset
        using the corresponding annotation

        Args:
            limit: int
                if there's a limit passed, only process up to this many annotations
        Returns:
            merged_data: dict
                If successful return the merged data from all source videos and annotations
        Raises:
            exception: VideoDataDataloopMergeServiceError

        """
        success = False
        if self.process_videos:
            success = self.generate_dataset_from_videos(limit=limit)
        else:
            success = self.generate_dataset_from_sequence_data(limit=limit)

        if success and write_to_file:
            self.write_merged_data_to_file()
        if not success:
            raise VideoDataDataloopMergeServiceError("Unable to create dataset")

        return self.merged_data

    def generate_dataset_from_sequence_data(self, limit: int = None):
        process_counter = 0
        for annotation, sequence in self.annotation_sequence_map.items():
            if limit and process_counter == limit:
                break
            video_data = None
            annotation_data = None
            with open(sequence) as f:
                video_data = json.load(f)
            with open(annotation) as f:
                annotation_data = json.load(f)
            segmented_data = self.transformer.segment_video_data_with_annotations(
                video_data=video_data, dataloop_data=annotation_data
            )
            self.merge_segmented_data(segmented_data=segmented_data)
            process_counter += 1
        return True

    def generate_dataset_from_videos(self, limit: int = None):
        vds = VideoDataService()
        process_counter = 0
        for annotation, video in self.annotation_video_map.items():
            if limit and process_counter == limit:
                break
            annotation_data = json.load(open(annotation))
            video_data = vds.process_video(
                input_filename=os.path.basename(video),
                video_input_path=os.path.split(video)[0],
                output_data_path=self.output_keypoints_path,
                output_keypoint_data_path=self.output_keypoints_path,
                write_keypoints_to_file=False,
                write_serialized_sequence_to_file=True,
                key_off_frame_number=False,
                configuration={},
            )
            segmented_data = self.transformer.segment_video_data_with_annotations(
                video_data=video_data, dataloop_data=annotation_data
            )
            self.merge_segmented_data(segmented_data=segmented_data)
            process_counter += 1
        return True

    def write_merged_data_to_file(self) -> bool:
        """
        This method is responsible for writing the contents of the merged data dictionary to a json file

        Returns:
            success: bool
                True if operation was successful
        """
        merged_data_json = json.dumps(self.merged_data, indent=4)
        path_utility.write_to_json_file(
            file_path=self.output_data_path,
            file_name=f"combined_dataset_{time.time_ns()}.json",
            data=merged_data_json,
        )
        return True

    def merge_segmented_data(self, segmented_data: dict) -> None:
        """
        This method takes a segmented data dictionary and merges it into the class's merged data dictionary

        Args:
            segmented_data: dict
                A dictionary keyed off annotation labels containing a list of frame data dictionaries representing all the frames for a labeled clip of video
        Raises:
            exception: VideoDataDataloopMergeServiceError
        """
        try:
            for label, examples in segmented_data.items():
                if (label not in self.merged_data) or (
                    not bool(self.merged_data[label])
                ):
                    self.merged_data[label] = []
                for example in examples:
                    self.merged_data[label].append(example)
        except:
            raise VideoDataDataloopMergeServiceError(
                "There was an issue merging segmented data"
            )


class VideoDataDataloopMergeServiceError(Exception):
    """Raised when there's a problem in the VideoDataDatloopMergeService class"""

    pass
