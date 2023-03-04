import os
from pose_parser.services.video_data_service import VideoDataService
from pose_parser.services.dataloop_annotation_transformer import (
    DataloopAnnotationTransformer,
)
from datetime import datetime

now = datetime.now()

# mm-dd-YY_H:M:S
dt_string = now.strftime("%m-%d-%Y_%H:%M:%S")
source_path = "/Volumes/NG External/ai_tango_video"


class ProcessVideosJob:
    @staticmethod
    def process_videos(source_path: str):
        for root, dir_names, file_names in os.walk(source_path):
            for f in file_names:
                if f.endswith("webm"):
                    pass
