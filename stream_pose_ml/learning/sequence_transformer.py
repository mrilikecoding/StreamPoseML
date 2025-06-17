from abc import ABC, abstractmethod

import pandas as pd
import numpy as np

import stream_pose_ml.services.segmentation_service as ss


class SequenceTransformer(ABC):
    @abstractmethod
    def transform(self, data: any, columns: list) -> any:
        """Transform the passed data into a row with the passed columns"""
        pass


from collections import defaultdict


# TODO create concrete classes for different schemes
class TenFrameFlatColumnAngleTransformer(SequenceTransformer):
    """This is the first transformer I'm writing here, so maybe a little more
    hard coded than I'd prefer. Just getting this working to close the interaction loop for now.

    TODO ideally there's some sort of schema generated from the dataset generation step
    that can be used to transform the sequence data back into the right columnnar format.
    So dataset generation is sequence_data->csv + schema, and then test example is sequence_data+schema->test_example

    """

    def transform(self, data: any, columns: list) -> any:
        # TODO -
        # take the data and pass into segmentation service
        # may need to alter some SegService strategy to
        # handle single row? Using Dataset / LabeledClip
        # models doesn't make sense, but as a static method
        # could work. Would like to share the functionality
        # between the generation of the dataset rows for
        # training and the building of an example from live
        # sequence data. They should have the same structure.
        # Further, it'd be ideal if when saving the trained
        # model, the schema of the data used is also saved
        # so that the schema can be used by the web app to
        # transform the input video data for use in classification
        #
        # ss.SegmentationService.flatten_into_columns()
        # Set top level keys from last frame
        frame_segment = data["frames"]
        flattened = {
            key: value
            for key, value in frame_segment[-1].items()
            if (isinstance(value, str) or value is None)
        }
        flattened["data"] = defaultdict(dict)
        for i, frame in enumerate(frame_segment):
            frame_items = frame.items()
            for key, value in frame_items:
                flattened_data_key = flattened["data"][key]
                if isinstance(value, dict):
                    value_items = value.items()
                    for k, v in value_items:
                        flattened_data_key[f"frame-{i+1}-{k}"] = v
                else:
                    flattened_data_key = value

        data = flattened["data"]
        output_dict = {"angles": data["angles"], "distances": data["distances"]}
        meta_keys = ["type", "sequence_id", "sequence_source", "image_dimensions"]
        output_meta = {key: flattened[key] for key in meta_keys}
        output_flattened = pd.json_normalize(data=output_dict)
        output_flattened_filtered = output_flattened.filter(columns)
        return (output_flattened_filtered, output_meta)
