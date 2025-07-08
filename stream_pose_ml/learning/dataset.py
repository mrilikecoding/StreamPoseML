from typing import Any


class Dataset:
    """All data for a dataset as well as a segmented representation of the data."""

    def __init__(
        self,
        all_frames: list[Any] | None = None,
        labeled_frames: list[Any] | None = None,
        unlabeled_frames: list[Any] | None = None,
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
        if unlabeled_frames is None:
            unlabeled_frames = []
        if labeled_frames is None:
            labeled_frames = []
        if all_frames is None:
            all_frames = []
        self.all_frames = all_frames
        self.labeled_frames = labeled_frames
        self.unlabeled_frames = unlabeled_frames
        self.segmented_data: list | None = None
