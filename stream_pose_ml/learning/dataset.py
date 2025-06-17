class Dataset:
    """All data for a dataset as well as a segmented representation of the data."""

    def __init__(
        self,
        all_frames: list = [],
        labeled_frames: list = [],
        unlabeled_frames: list = [],
    ) -> None:
        """Init a Dataset object.


        Args:
            all_frames: list
                all frame data
            labeled_frames: list
                only labeled frames
            unlabeled_frames: list
                only unlabeled frames
        """
        self.all_frames = all_frames
        self.labeled_frames = labeled_frames
        self.unlabeled_frames = unlabeled_frames
        self.segmented_data = None
