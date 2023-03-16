import pandas as pd
import time

from pose_parser.services.video_data_dataloop_merge_service import (
    VideoDataDataloopMergeService,
)

from pose_parser.services.segmentation_service import SegmentationService
from pose_parser.learning.dataset import Dataset
from pose_parser.serializers.dataset_serializer import DatasetSerializer


def round_nested_dict(item: dict, precision: int = 4):
    """This method takes a dictionary and recursively rounds float values to the indicated precision

    Args:
        item: dict
            A dictionary with nested keys and floats that need to be rounded
        precision: int
            How many decimals to round to
    """
    if isinstance(item, dict):
        return type(item)(
            (key, round_nested_dict(value, precision)) for key, value in item.items()
        )
    if isinstance(item, float):
        return round(item, precision)
    return item


class BuildAndFormatDatasetJob:
    """This class works through json sequence data and annotation data to compile a dataset"""

    @staticmethod
    def build_dataset_from_data_files(
        annotations_data_directory: str,
        sequence_data_directory: str,
        limit: int | None = None,
    ):
        """This method builds a dataset from pre-processed video data and annotations.

        Args:
            annotations_data_directory: str
                location of annotations corresponding to video data json
            sequence_data_directory: str
                location of serialized sequence data files (this is output of BlazePoseSequenceSerializer written to json)
            limit: int
                only process data to this limit (useful for testing)
        """
        vdms = VideoDataDataloopMergeService(
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
        """This method builds a dataset and also does the video processing for all videos within a directory.

        Use this when you want to go directly from source videos and annotations to a dataset.

        Args:
            annotations_directory: str
                the path to annotations
            video_directory: str
                the location of videos to be processed
        """
        vdms = VideoDataDataloopMergeService(
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
        segmentation_strategy: str | None = None,
        segmentation_window: int | None = None,
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
            segmentation_window: int | None
                if segmentation strategy is "window" this will be the frame window size

        """
        segmentation_service = SegmentationService(
            include_unlabeled_data=include_unlabeled_data,
            segmentation_strategy=segmentation_strategy,
            segmentation_window=segmentation_window,
        )
        segmented_dataset = segmentation_service.segment_dataset(dataset)
        dataset_serializer = DatasetSerializer(pool_rows=pool_frame_data_by_clip)
        formatted_data = dataset_serializer.serialize(segmented_dataset)

        if decimal_precision is not None:
            rounded = []
            for row in formatted_data:
                rounded_row = round_nested_dict(item=row, precision=4)
                rounded.append(rounded_row)
            formatted_data = rounded

        return formatted_data

    @staticmethod
    def write_dataset_to_csv(csv_location: str, formatted_dataset: list):
        """Write the passed serialized dataset to a csv.

        This method will flatten the passed json and save to a timestamped file.

        Args:
            csv_location: str
                path to where file should be saved
            formatted_dataset: list[dict]
                list of serialized data dicts
        Returns:
            success: bool
                True if successful
        """
        df = pd.json_normalize(data=formatted_dataset)
        filename = f"dataset_{time.time_ns()}.csv"
        output_path = f"{csv_location}/{filename}"
        df.to_csv(output_path)
        return True


class BuildAndFormatDatasetJobError(Exception):
    """Raise when there's an issue with the BuildAndFormatDatasetJob class"""
