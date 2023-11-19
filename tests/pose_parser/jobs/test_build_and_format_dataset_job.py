import unittest
import yaml

CONFIG = yaml.safe_load(open("./config.yml"))

from stream_pose_ml.jobs.build_and_format_dataset_job import BuildAndFormatDatasetJob


class TestBuildAndFormatDatasetJob(unittest.TestCase):
    def setUp(self) -> None:
        self.sequence_data_directory = CONFIG["test_sequence_data_output_directory"]
        self.annotation_data_directory = CONFIG["test_annotations_directory"]
        self.merged_annotation_output_directory = CONFIG[
            "test_merged_annotations_output_directory"
        ]
        return super().setUp()

    def test_build_dataset_from_data_files(self):
        annotations_data_directory = self.annotation_data_directory
        sequence_data_directory = self.sequence_data_directory
        merged_annotation_output_directory = self.merged_annotation_output_directory

        # Uncomment this to test against real files - TODO need a nice way to do this :)
        sequence_data_directory = CONFIG["sequence_data_output_directory"]
        annotations_data_directory = CONFIG["source_annotations_directory"]
        merged_annotation_output_directory = CONFIG[
            "merged_annotations_output_directory"
        ]

        data_builder = BuildAndFormatDatasetJob()
        dataset = data_builder.build_dataset_from_data_files(
            annotations_data_directory=annotations_data_directory,
            sequence_data_directory=sequence_data_directory,
            limit=None,
        )
        formatted_dataset = data_builder.format_dataset(
            dataset=dataset,
            pool_frame_data_by_clip=True,
            decimal_precision=4,
            include_unlabeled_data=True,
            segmentation_strategy="flatten_into_columns",
            segmentation_splitter_label="step_type",
            segmentation_window=25,
            segmentation_window_label="weight_transfer_type",
        )
        data_builder.write_dataset_to_csv(
            csv_location=merged_annotation_output_directory,
            formatted_dataset=formatted_dataset,
        )
