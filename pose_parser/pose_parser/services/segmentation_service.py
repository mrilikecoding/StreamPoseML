import typing
from enum import Enum

if typing.TYPE_CHECKING:
    from pose_parser.learning.dataset import Dataset

from pose_parser.learning.labeled_clip import LabeledClip


class SegmentationStrategy(Enum):
    """This class enumerates different strategies for segmenting video frame data"""

    SPLIT_ON_LABEL = "split_on_label"
    WINDOW = "window"
    NONE = "none"


class SegmentationService:
    """Given a list of data, devide into various types of segmented data.

    A segment of data should correspond to a training example. This allows
    specifying different strategies for taking a Dataset's frame data, which
    has lists of labeled/unlabeled data for every video, and parsing it out into
    different training examples that aren't necessarily dependent on the originating video.

    For example, if segmentation strategy is "split_by_label", then the segmented data
    will look like a list of segments that represent a sequence of frames sharing a label.

    So a list of list of frames (f1,f2...) from different videos segmented by L/R would go from:
    video1: [f1-L, f2-L, f3-R, f4-R] video2:[f5-L, f6-R, f7-R]]
    To:
    segmented_data: [[f1-L, f2-L], [f3-R, f4-R], [f5-L], [f6-R, f7-R]]

    """

    segemetation_strategy: str
    include_unlabeled_data: bool
    segmentation_window: int

    def __init__(
        self,
        segmentation_strategy: str,
        include_unlabeled_data: bool = False,
        segmentation_window: int | None = None,
    ) -> None:
        """Init a SegmentationService object.

        Args:
            segmentation_strategy: str
                A string value corresponding to an enumerated SegmentationStrategy type.
                This will determine how video frames are aggregated.
                    "none" - split every frame into its own segment/clip/example



        """
        self.segemetation_strategy = SegmentationStrategy(segmentation_strategy)
        self.segmentation_window = segmentation_window
        self.include_unlabeled_data = include_unlabeled_data
        self.merged_data = []

    def segment_dataset(self, dataset: "Dataset") -> "Dataset":
        segmented_data = None
        if self.segemetation_strategy == SegmentationStrategy.NONE:
            segmented_data = self.segment_all_frames(dataset=dataset)
        elif self.segemetation_strategy == SegmentationStrategy.SPLIT_ON_LABEL:
            segmented_data = self.split_on_label(dataset=dataset)

        dataset.segmented_data = segmented_data
        return dataset

    def segment_all_frames(self, dataset: "Dataset") -> list[LabeledClip]:
        """This method creates a list of where every frame is its own LabeledClip.

        When the segmentation strategy is "none", each frame's data is its own training example.
        So a "clip" is created corresponding to each frame, so each clip has a frame length of 1.

        Args:
            dataset: Dataset
                a Dataset object instantiated with attributes "all_frames", "labeled_frames", and "unlabeled_frames"
        Returns:
            segmented_data: list[LabeledClip]
                a list of LabeledClip objects corresponding to every frame
        """
        segmented_data = []
        if self.include_unlabeled_data:
            for frame in sum(dataset.all_frames, []):
                segmented_data.append(LabeledClip(frames=[frame]))
        else:
            for frame in sum(dataset.labeled_frames, []):
                segmented_data.append(LabeledClip(frames=[frame]))
        return segmented_data

    def split_on_label(
        self, dataset: "Dataset", segment_splitter_label: str
    ) -> list[list[dict]]:
        """Split a list of dicts into a list of lists of dicts representing segments of data.

        Args:
            segment_splitter_label: str
                start a new segment when there's a change in this label's value from frame to frame
            dataset: Dataset
                a Dataset object
        Returns:
            segmented_data
        """
        pass
        # segmented_frames = {}
        # segment_counter = 0
        # for video in labeled_frames:
        #     for i, frame in enumerate(labeled_frames):
        #         # if this is the last frame don't compare to next
        #         if (i + 1) == len(labeled_frames):
        #             if segment_counter in segmented_frames:
        #                 segmented_frames[segment_counter].append(frame)
        #             else:
        #                 segmented_frames[segment_counter] = [frame]
        #         elif (
        #             labeled_frames[i + 1][segment_splitter_label]
        #             == frame[segment_splitter_label]
        #         ):
        #             if segment_counter in segmented_frames:
        #                 segmented_frames[segment_counter].append(frame)
        #             else:
        #                 segmented_frames[segment_counter] = [frame]
        #         else:
        #             segment_counter += 1
        #             if segment_counter in segmented_frames:
        #                 segmented_frames[segment_counter].append(frame)
        #             else:
        #                 segmented_frames[segment_counter] = [frame]

        # segmented_data = list(segmented_frames.values())
        # if unlabeled_frames is not None:
        #     segmented_data.append(unlabeled_frames)
        # return segmented_data

    def split_on_window(self):
        pass
