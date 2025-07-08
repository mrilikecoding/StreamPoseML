class LabeledClip:
    """A collection frames that represent a labeled sequence"""

    frames: list[dict]
    sequence_length: int

    def __init__(self, frames: list[dict] | None = None):
        if frames is None:
            frames = []
        self.frames = frames
        self.sequence_length = len(frames)
        # TODO make LabeledFrames out of the frame data
