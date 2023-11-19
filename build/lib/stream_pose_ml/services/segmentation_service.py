import typing
from enum import Enum

if typing.TYPE_CHECKING:
    from stream_pose_ml.learning.dataset import Dataset

from stream_pose_ml.learning.labeled_clip import LabeledClip


class SegmentationStrategy(Enum):
    """This class enumerates different strategies for segmenting video frame data"""

    SPLIT_ON_LABEL = "split_on_label"
    FLATTEN_INTO_COLUMNS = "flatten_into_columns"
    FLATTEN_ON_EXAMPLE = "flatten_on_example"
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
        """Segment the data in a dataset into various training examples.

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
        elif self.segemetation_strategy == SegmentationStrategy.FLATTEN_INTO_COLUMNS:
            dataset.segmented_data = self.flatten_into_columns(dataset=dataset)
        elif self.segemetation_strategy == SegmentationStrategy.FLATTEN_ON_EXAMPLE:
            dataset.segmented_data = self.flatten_on_example(dataset=dataset)

        return dataset

    @staticmethod
    def flatten_segment_into_row(frame_segment: list):
        """Flatten a list of frames into a single row object

        For this scheme want to store nested joint / angle / distance data per frame
        in top level keys for consistent serialization
        so here, restructuring frame list data into a flattened
        column representation where original shape is preserved but internally
        new keys are created for each individual frame

        Args:
            frame_segment: list
                A list of video frames

        Returns:
            flattened_segment: dict
                a single object representing the flattened frames

        """
        # Set top level keys from last frame
        flattened = {
            key: value
            for key, value in frame_segment[-1].items()
            if (isinstance(value, str) or value is None)
        }
        flattened["data"] = {}
        # Set internal data keys to the same top level keys
        for i, frame in enumerate(frame_segment):
            frame_data = frame["data"]
            for key, value in frame_data.items():
                if key not in flattened["data"]:
                    flattened["data"][key] = {}
                # Merge all frame data for this segment into frame specific keys
                if isinstance(value, dict):
                    for k, v in value.items():
                        flattened["data"][key][f"frame-{i+1}-{k}"] = v
                else:
                    # Let the last frame set the top level value here
                    # when we don't have nested data
                    flattened["data"][key] = value

        return flattened

    def flatten_on_example(self, dataset: "Dataset") -> list[LabeledClip]:
        """Segment video frame data based on a fixed window size from the end of a complete labeled example and flatten the data into frame columns

        NOTE you'll likely want to keep the frame window small, otherwise there will be MANY columns of data.

        This is basically a combo of flatten into columns and split on label.
        The clip is split on label and then based on the window size the frames within the segment are flattened into a single row of frame-based columns

        Args:
            dataset: Dataset
                a Dataset with frame data
        Returns:
            labeled_clips: list[LabeledClip]
                a list of one LabeledClip object where the clip represents
                a window of data where the last frame is labeled and internal data
                stores keys for each invidual frame in the sequence

        """
        return self.split_on_label(dataset=dataset, flatten_into_columns=True)

    def flatten_into_columns(self, dataset: "Dataset") -> list[LabeledClip]:
        """Segment video frame data based on a fixed window size and flatten the data into frame columns

        NOTE you'll likely want to keep the frame window small, otherwise there will be MANY columns of data.

        For each video, find a specified number of frames where the last frame has a certain label
        and use as a training example. Then flatten the frame data into frame specific columns.
        This is very similar to the frame window, except rather than the list of clips representing
        all frames, here the "clip" is a single representation where top level data keys like
        "angles", "joints", and "distances" internally have the metrics keyed off every frame in the
        segment. This allows for a single high-dimensional representation of the data within a
        fixed sequence of frames.

        Args:
            dataset: Dataset
                a Dataset with frame data
        Returns:
            labeled_clips: list[LabeledClip]
                a list of one LabeledClip object where the clip represents
                a window of data where the last frame is labeled and internal data
                stores keys for each invidual frame in the sequence

        """
        if self.segmentation_window is None or self.segmentation_window_label is None:
            raise SegmentationServiceError(
                'Both segmentation window and segmentation window label is required for segmentation strategy "flatten_into_columns".'
            )
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

                    # Set top level keys from last frame
                    flattened = self.flatten_segment_into_row(
                        frame_segment=frame_segment
                    )

                    segmented_frames[segment_counter] = [flattened]
                    segment_counter += 1

            segmented_data = list(segmented_frames.values())
            segmented_video_data_list.append(segmented_data)
        merged_segmented_data = sum(segmented_video_data_list, [])
        labeled_clips = [
            LabeledClip(frames=example_frames)
            for example_frames in merged_segmented_data
        ]
        return labeled_clips

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
        self, dataset: "Dataset", flatten_into_columns: bool = False
    ) -> list[LabeledClip]:
        """Split a Datasets labeled_frame list into a list of lists of dicts representing segments of data.

        The Dataset object should have a list of lists of frames where each list is separated by source video.
        This function will reformat this list into a list of lists of frames sharing the same label (segment_splitter_label).
        This translates to list of training example video clips.

        Args:
            segment_splitter_label: str
                start a new segment when there's a change in this label's value from frame to frame
            dataset: Dataset
                a Dataset object
            flatten_into_columns: bool
                whether to flatten the frames into a single row of frame-based feature columns
        Returns:
            labeled_clips: list[LabeledClip]
                a list of LabelClip objects where each LabeledClip
                is a sequence of data sharing the same segment_splitter_label.
                this means that each LabeledClip represents an example of training data
        """
        if self.segmentation_splitter_label is None:
            raise SegmentationServiceError(
                'segmentation_spliiter_label must be present for segmentation strategy "split_on_label".'
            )
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
            # if there's a segmentation window, use it to only use the last x frames from each example
            if self.segmentation_window:
                segmented_data = [
                    segment[-self.segmentation_window :] for segment in segmented_data
                ]
                if flatten_into_columns:
                    segmented_data = [
                        [self.flatten_segment_into_row(segment)]
                        for segment in segmented_data
                    ]

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
        if self.segmentation_window is None or self.segmentation_window_label is None:
            raise SegmentationServiceError(
                'Both segmentation window and segmentation window label is required for segmentation strategy "split_on_window".'
            )
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


class SegmentationServiceError(Exception):
    """Raise when there's an issue in the Segmentation Service"""

    pass
