class LabeledFrame:
    """Labeled frame data"""

    def __init__(
        self,
        frame_data: dict,
        video_frame_number: int,
        clip_frame_number: int,
        video_id: str = None,
        labels: list[str] | None = None,
    ) -> None:
        self.frame_data = frame_data
        self.video_frame_number = video_frame_number

        self.clip_frame_number = clip_frame_number  # step frame id
        self.video_id = video_id
        self.labels = labels
        self.label_data = {}
