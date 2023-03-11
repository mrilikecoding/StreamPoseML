from pose_parser.services.video_data_dataloop_merge_service import (
    VideoDataDataloopMergeService,
)
from pose_parser.services.dataset_output_transformer_service import (
    DatasetOutputTransformerService,
)


class BuildAndFormatDatasetJob:
    """This class works through json sequence data and annotation data to compile a dataset"""

    @staticmethod
    def build_dataset_from_data_files(
        annotations_data_directory: str,
        sequence_data_directory: str,
        merged_dataset_path: str | None = None,
        write_to_file: bool = False,
        limit: int | None = None,
        opts: dict = {},
    ):
        vdms = VideoDataDataloopMergeService(
            annotations_data_directory=annotations_data_directory,
            sequence_data_directory=sequence_data_directory,
            process_videos=False,
            output_data_path=merged_dataset_path,
        )

        # TODO - write to file is too difficult with data this big
        # 5 videos resulted in a 235 mb json file. Yikes!
        dataset = vdms.generate_dataset(limit=limit, write_to_file=write_to_file)

        dots = DatasetOutputTransformerService(opts=opts)
        dots.format_dataset(generated_raw_dataset=dataset)
        pass

    def build_dataset_from_videos(
        annotations_directory: str,
        video_directory: str,
        write_to_file: bool = False,
        limit: int | None = None,
    ):
        vdms = VideoDataDataloopMergeService(
            annotations_directory=annotations_directory,
            video_directory=video_directory,
            process_videos=True,
        )

        vdms.generate_dataset(limit=limit, write_to_file=write_to_file)
