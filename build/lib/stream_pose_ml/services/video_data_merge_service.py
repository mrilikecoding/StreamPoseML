import os
import json
from stream_pose_ml.utils import path_utility

from stream_pose_ml.services.annotation_transformer_service import (
    AnnotationTransformerService,
)


from stream_pose_ml.learning.labeled_clip import LabeledClip
from stream_pose_ml.services import video_data_service as vds


class VideoDataMergeService:
    """Merge annotations with video data.

    This class is responsible for searching an annotations directory to find
    relevant annotations for a specified video and combining it with right video data.
    """

    annotations_data_directory: str
    output_keypoints_path: str
    sequence_data_directory: str
    process_videos: bool
    video_directory: str

    def __init__(
        self,
        annotations_data_directory: str,
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
            output_keypoints_path: str
                where to put keypoint data
        """
        self.annotations_data_directory = annotations_data_directory
        self.video_directory = video_directory
        self.sequence_data_directory = sequence_data_directory
        self.output_keypoints_path = (output_keypoints_path,)
        self.annotation_video_map = {}
        self.video_annotation_map = {}
        self.annotation_sequence_map = {}
        self.sequence_annotation_map = {}
        self.process_videos = process_videos
        self.merged_data = []

        self.transformer = AnnotationTransformerService()

        self.create_video_annotation_map()

    def create_video_annotation_map(self):
        """This method reads from the annotations and videos directory to
        create a file path map between videos and annotations.

        Returns:
            success: bool
                Returns: True if operation completes successfully

        Raises:
            exception: VideoDataMergeServiceError
                Raises:d if there's an error creating this map
        """
        if self.process_videos and self.video_directory is None:
            raise VideoDataMergeServiceError(
                "There is no source video directory specified to generate video data from."
            )

        annotation_files = path_utility.get_file_paths_in_directory(
            directory=self.annotations_data_directory, extension="json"
        )

        if self.video_directory:
            # TODO maybe makes sense to set this in config?
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

    def generate_annotated_video_data(
        self,
        limit: int = None,
    ) -> dict:
        """Use this object's generated map to create a nested dataset

        This method is reponsible for calling the video data service
        based on the map and merging the returned data into a dataset
        using the corresponding annotation

        Args:
            limit: int
                if there's a limit passed, only process up to this many annotations
        Returns:
            annotated_data: dict[str, list[dict]]
                If successful return the merged annotated data from all source videos and annotations
                {"all_frames": [...], "labeled_frames": [...], "unlabeled_frames": [...]}
        Raises:
            exception: VideoDataMergeServiceError

        """
        if self.process_videos:
            self.generate_video_data_from_videos(limit=limit)
        else:
            self.generate_video_data_from_sequence_data(limit=limit)

        merged_all_frames = []
        merged_labeled_frames = []
        merged_unlabeled_frames = []
        merged_video_data = [data["video_data"] for data in self.merged_data]
        merged_annotation_data = [data["annotation_data"] for data in self.merged_data]
        for video_data, annotation_data in zip(
            merged_video_data, merged_annotation_data
        ):
            (
                all_frames,
                labeled_frames,
                unlabeled_frames,
            ) = self.transformer.update_video_data_with_annotations(
                video_data=video_data, annotation_data=annotation_data
            )
            merged_all_frames.append(all_frames)
            merged_labeled_frames.append(labeled_frames)
            merged_unlabeled_frames.append(unlabeled_frames)

        annotated_data = {
            "all_frames": merged_all_frames,
            "labeled_frames": merged_labeled_frames,
            "unlabeled_frames": merged_unlabeled_frames,
        }
        return annotated_data

    def generate_video_data_from_sequence_data(
        self, limit: int = None
    ) -> tuple[dict, dict]:
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
            process_counter += 1
            self.merged_data.append(
                {"video_data": video_data, "annotation_data": annotation_data}
            )
        return True

    def generate_video_data_from_videos(self, limit: int = None) -> tuple[dict, dict]:
        video_data_service = vds.VideoDataService()
        process_counter = 0
        for annotation, video in self.annotation_video_map.items():
            if limit and process_counter == limit:
                break
            annotation_data = json.load(open(annotation))
            video_data = video_data_service.process_video(
                input_filename=os.path.basename(video),
                video_input_path=os.path.split(video)[0],
                output_data_path=self.output_keypoints_path,
                output_keypoint_data_path=self.output_keypoints_path,
                write_keypoints_to_file=False,
                write_serialized_sequence_to_file=True,
                key_off_frame_number=False,
                configuration={},
            )
            process_counter += 1
        self.merged_data.append(
            {"video_data": video_data, "annotation_data": annotation_data}
        )
        return True


class VideoDataMergeServiceError(Exception):
    """Raised when there's a problem in the VideoDataDatloopMergeService class"""

    pass
