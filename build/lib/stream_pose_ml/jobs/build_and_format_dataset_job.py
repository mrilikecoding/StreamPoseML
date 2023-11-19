import pandas as pd
import time

from stream_pose_ml.services.video_data_merge_service import (
    VideoDataMergeService,
)

from stream_pose_ml.utils.utils import round_nested_dict
from stream_pose_ml.services.segmentation_service import SegmentationService
from stream_pose_ml.learning.dataset import Dataset
from stream_pose_ml.serializers.dataset_serializer import DatasetSerializer


class BuildAndFormatDatasetJob:
    """Work through json sequence data and annotation data to compile a dataset"""

    @staticmethod
    def build_dataset_from_data_files(
        annotations_data_directory: str,
        sequence_data_directory: str,
        limit: int | None = None,
    ):
        """Build a dataset from pre-processed video data and annotations.

        Args:
            annotations_data_directory: str
                location of annotations corresponding to video data json
            sequence_data_directory: str
                location of serialized sequence data files (this is output of BlazePoseSequenceSerializer written to json)
            limit: int
                only process data to this limit (useful for testing)
        """
        vdms = VideoDataMergeService(
            annotations_data_directory=annotations_data_directory,
            sequence_data_directory=sequence_data_directory,
            process_videos=False,
        )

        annotated_video_data = vdms.generate_annotated_video_data(limit=limit)
        # TODO rename these attrs all_frames_raw or something to distinguish from segmented data...
        dataset = Dataset(
            all_frames=annotated_video_data["all_frames"],
            labeled_frames=annotated_video_data["labeled_frames"],
            unlabeled_frames=annotated_video_data["unlabeled_frames"],
        )
        return dataset

    @staticmethod
    def build_dataset_from_videos(
        annotations_directory: str,
        video_directory: str,
        limit: int | None = None,
    ):
        """Builds a dataset while also doing the video processing for all videos within a directory.

        Use this when you want to go directly from source videos and annotations to a dataset.

        Args:
            annotations_directory: str
                the path to annotations
            video_directory: str
                the location of videos to be processed
        """
        vdms = VideoDataMergeService(
            annotations_directory=annotations_directory,
            video_directory=video_directory,
            process_videos=True,
        )

        annotated_video_data = vdms.generate_annotated_video_data(limit=limit)
        dataset = Dataset(
            all_frames=annotated_video_data["all_frames"],
            labeled_frames=annotated_video_data["labeled_frames"],
            unlabeled_frames=annotated_video_data["unlabeled_frames"],
        )
        return dataset

    @staticmethod
    def format_dataset(
        dataset: Dataset,
        pool_frame_data_by_clip: bool = True,
        decimal_precision: int | None = None,
        include_unlabeled_data: bool = False,
        include_angles: bool = True,
        include_distances: bool = True,
        include_normalized: bool = True,
        include_joints: bool = False,
        include_z_axis: bool = False,
        segmentation_strategy: str | None = None,
        segmentation_splitter_label: str | None = None,
        segmentation_window: int | None = None,
        segmentation_window_label: str | None = None,
    ):
        """Serialize a list of dataset clip data

        Args:
            dataset: list
                a list of LabeledClip objects
            group_frames_by_clip: bool
                When True, the returned dataset will pool frame data across all frames
                When False the returned dataset will return each frame as a separate row
            decimal_precision: int | None
                if decimal precision is specified, round all float values in dataset to this number of places
            include_unlabeled_data: bool
                depending on how the video frame data is segmented into clips it may or may not be useful to have unlabeled frame data
                using a temporal window, likely you'll want segments that include unlabeled frames as long as the last frame is labeled
            segmentation_strategy: str | None
                one of "split_on_label", "window", "none"
            segmentation_splitter_label: str | None
                if segmentation strategy is "split_on_label" this will be the label used to segment data into training examples.
            segmentation_window: int | None
                if segmentation strategy is "window" this will be the frame window size
            segmentation_window_label: str | None
                if segmentation strategy is "window" this will be the label used to segment data into training examples.

        """
        segmentation_service = SegmentationService(
            include_unlabeled_data=include_unlabeled_data,
            segmentation_strategy=segmentation_strategy,
            segmentation_splitter_label=segmentation_splitter_label,
            segmentation_window=segmentation_window,
            segmentation_window_label=segmentation_window_label,
        )
        segmented_dataset = segmentation_service.segment_dataset(dataset)
        dataset_serializer = DatasetSerializer(
            pool_rows=pool_frame_data_by_clip,
            include_normalized=include_normalized,
            include_angles=include_angles,
            include_distances=include_distances,
            include_joints=include_joints,
            include_z_axis=include_z_axis,
        )
        formatted_data = dataset_serializer.serialize(segmented_dataset)

        if decimal_precision is not None:
            rounded = []
            for row in formatted_data:
                rounded_row = round_nested_dict(item=row, precision=4)
                rounded.append(rounded_row)
            formatted_data = rounded

        return formatted_data

    @staticmethod
    def write_dataset_to_csv(
        csv_location: str, formatted_dataset: list, filename: str = None
    ):
        """Write the passed serialized dataset to a csv.

        This method will flatten the passed json and save to a timestamped file.

        Args:
            csv_location: str
                path to where file should be saved
            formatted_dataset: list[dict]
                list of serialized data dicts
            filename: str
                if a custom filename is desired, pass in here. otherwise this will be a timestamp
        Returns:
            success: bool
                True if successful
        """
        df = pd.json_normalize(data=formatted_dataset)
        if filename is None:
            filename = f"dataset_{time.time_ns()}"
        output_path = f"{csv_location}/{filename}.csv"
        df.to_csv(output_path)
        return True


class BuildAndFormatDatasetJobError(Exception):
    """Raise when there's an issue with the BuildAndFormatDatasetJob class"""
