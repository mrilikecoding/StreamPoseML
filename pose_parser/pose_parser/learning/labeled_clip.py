class LabeledClip:
    def __init__(self, labels: list[str], frames: list[dict]):
        self.labels = labels
        self.frames = frames
        self.sequence_length = len(frames)