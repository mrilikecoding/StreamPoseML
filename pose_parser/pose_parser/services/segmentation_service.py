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

    TODO should this be merged into the Dataset class?
    """

    segemetation_strategy: str
    include_unlabeled_data: bool
    segmentation_window: int

    def __init__(
        self,
        segmentation_strategy: str,
        include_unlabeled_data: bool = False,
        segmentation_window: int | None = None,
        segmentation_splitter_label: str | None = None,
        segmentation_window_label: str | None = None,
    ) -> None:
        """Init a SegmentationService object.

        Args:
            segmentation_strategy: str
                A string value corresponding to an enumerated SegmentationStrategy type.
                This will determine how video frames are aggregated.
                    "none" - split every frame into its own segment/clip/example
                    "split_on_label" - segment frames into examples based on frames grouped by label (in sequence)
                        i.e. a sequence labeled [[L,L,L,R,R,R,L,L,L]] would result in [[L, L, L], [R, R, R], [L, L, L]]
        """
        # TODO pass this in
        self.segemetation_strategy = SegmentationStrategy(segmentation_strategy)
        self.segmentation_window = segmentation_window
        self.segmentation_splitter_label = segmentation_splitter_label
        self.segmentation_window_label = segmentation_window_label
        self.include_unlabeled_data = include_unlabeled_data
        self.merged_data = []

    def segment_dataset(self, dataset: "Dataset") -> "Dataset":
        """ Segment the data in a dataset into various training examples.

        Using the a specified strategy on the class instance, generate LabeledClips to store on the dataset.

        Args:
            dataset: Dataset
                a Dataset object with data
        Returns:
            dataset: Dataset
                a Dataset with segmented data added to it based on user specified strategy / params
        """
        if self.segemetation_strategy == SegmentationStrategy.NONE:
            dataset.segmented_data = self.segment_all_frames(dataset=dataset)
        elif self.segemetation_strategy == SegmentationStrategy.SPLIT_ON_LABEL:
            dataset.segmented_data = self.split_on_label(dataset=dataset)
        elif self.segemetation_strategy == SegmentationStrategy.WINDOW:
            dataset.segmented_data = self.split_on_window(dataset=dataset)

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

    def split_on_label(self, dataset: "Dataset") -> list[LabeledClip]:
        """Split a Datasets labeled_frame list into a list of lists of dicts representing segments of data.

        The Dataset object should have a list of lists of frames where each list is separated by source video.
        This function will reformat this list into a list of lists of frames sharing the same label (segment_splitter_label).
        This translates to list of training example video clips.

        Args:
            segment_splitter_label: str
                start a new segment when there's a change in this label's value from frame to frame
            dataset: Dataset
                a Dataset object
        Returns:
            labeled_clips: list[LabeledClip]
                a list of LabelClip objects where each LabeledClip
                is a sequence of data sharing the same segment_splitter_label.
                this means that each LabeledClip represents an example of training data
        """
        labeled_frame_videos = dataset.labeled_frames
        segment_splitter_label = self.segmentation_splitter_label
        segmented_video_data_list = []
        for video in labeled_frame_videos:
            segmented_frames = {}
            segment_counter = 0
            for i, frame in enumerate(video):
                # if this is the last frame don't compare to next
                if (i + 1) == len(video):
                    if segment_counter in segmented_frames:
                        segmented_frames[segment_counter].append(frame)
                    else:
                        segmented_frames[segment_counter] = [frame]
                elif (
                    video[i + 1][segment_splitter_label]
                    == frame[segment_splitter_label]
                ):
                    if segment_counter in segmented_frames:
                        segmented_frames[segment_counter].append(frame)
                    else:
                        segmented_frames[segment_counter] = [frame]
                else:
                    segment_counter += 1
                    if segment_counter in segmented_frames:
                        segmented_frames[segment_counter].append(frame)
                    else:
                        segmented_frames[segment_counter] = [frame]

            segmented_data = list(segmented_frames.values())
            segmented_video_data_list.append(segmented_data)
        merged_segmented_data = sum(segmented_video_data_list, [])
        labeled_clips = [
            LabeledClip(frames=example_frames)
            for example_frames in merged_segmented_data
        ]
        return labeled_clips

    def split_on_window(self, dataset: "Dataset") -> list[LabeledClip]:
        """Segment video frame data based on a fixed window size.

        For each video, find a specified number of frames where the last frame has a certain label
        and use as a training example.

        Args:
            dataset: Dataset
                a Dataset with frame data
        Returns:
            labeled_clips: list[LabeledClip]
                a list of LabeledClip objects where each clip represents
                a window of data where the last frame is labeled

        """
        segment_window_size = self.segmentation_window
        segment_window_label = self.segmentation_window_label
        all_frame_videos = dataset.all_frames
        segmented_video_data_list = []
        for video in all_frame_videos:
            segmented_frames = {}
            segment_counter = 0
            for i, frame in enumerate(video):
                if i < segment_window_size:
                    continue
                elif (
                    i % segment_window_size == 0
                    and video[i][segment_window_label] is not None
                ):
                    frame_segment = []
                    for j in range(1 + i - segment_window_size, i + 1):
                        frame_segment.append(video[j])
                    segmented_frames[segment_counter] = frame_segment
                    segment_counter += 1

            segmented_data = list(segmented_frames.values())
            segmented_video_data_list.append(segmented_data)
        merged_segmented_data = sum(segmented_video_data_list, [])
        labeled_clips = [
            LabeledClip(frames=example_frames)
            for example_frames in merged_segmented_data
        ]
        return labeled_clips
