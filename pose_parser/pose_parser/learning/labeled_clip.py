class LabeledClip:
    """Object represeting a collection of frames"""

    frames: list[dict]
    sequence_length: int

    def __init__(self, frames: list[dict] = []):
        self.frames = frames
        self.sequence_length = len(frames)
        # TODO make LabeledFrames out of the frame data
