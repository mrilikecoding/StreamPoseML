from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd


class SequenceTransformer(ABC):
    """
    Abstract base class for sequence data transformers.

    TODO: Improve interface with better type annotations and documentation.
    TODO: Define a clear data structure schema for inputs and outputs.
    TODO: Add validation methods for input data structures.
    """

    @abstractmethod
    def transform(
        self, data: Any, columns: list[str]
    ) -> tuple[pd.DataFrame, dict] | tuple[dict, dict]:
        """
        Transform the passed data into a row with the passed columns.

        Args:
            data: Input data containing sequence information
            columns: List of column names to include in the output

        Returns:
            Tuple containing:
            - Either a DataFrame or Dict (depending on implementation) with
              transformed data
            - Dict with metadata about the transformation
        """
        pass


# TODO create concrete classes for different schemes
class TenFrameFlatColumnAngleTransformer(SequenceTransformer):
    """This is the first transformer I'm writing here, so maybe a little more
    hard coded than I'd prefer. Just getting this working to close the interaction
    loop for now.

    TODO ideally there's some sort of schema generated from the dataset generation
    step that can be used to transform the sequence data back into the right
    columnnar format. So dataset generation is sequence_data->csv + schema, and
    then test example is sequence_data+schema->test_example

    """

    def transform(self, data: Any, columns: list) -> Any:
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
        flattened: dict[str, Any] = {
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
                        flattened_data_key[f"frame-{i + 1}-{k}"] = v
                else:
                    flattened_data_key = value

        data = flattened["data"]
        output_dict = {"angles": data["angles"], "distances": data["distances"]}
        meta_keys = ["type", "sequence_id", "sequence_source", "image_dimensions"]
        output_meta = {key: flattened[key] for key in meta_keys}
        output_flattened = pd.json_normalize(data=output_dict)
        output_flattened_filtered = output_flattened.filter(columns)
        return (output_flattened_filtered, output_meta)


# TODO create concrete classes for different schemes
class MLFlowTransformer(SequenceTransformer):
    def transform(self, data: Any, columns) -> Any:
        # TDOD this def needs a refactor, but working for now
        frame_segment = data["frames"]

        # Flatten the frame data with default 0.0 where values are missing
        flattened: dict[str, Any] = {
            key: value if value is not None else 0.0  # default to 0.0 if value is None
            for key, value in frame_segment[-1].items()
            if isinstance(value, str | type(None))
        }

        flattened["data"] = defaultdict(dict)
        for i, frame in enumerate(frame_segment):
            for key, value in frame.items():
                flattened_data_key = flattened["data"][key]

                if isinstance(value, dict):
                    for k, v in value.items():
                        flattened_data_key[f"frame-{i + 1}-{k}"] = (
                            v if v is not None else 0.0
                        )  # default to 0.0 if None
                else:
                    flattened_data_key = (
                        value if value is not None else 0.0
                    )  # default to 0.0 if None

        data = flattened["data"]
        output_dict = {"joints": data["joint_positions"]}
        meta_keys = ["type", "sequence_id", "sequence_source", "image_dimensions"]
        output_meta = {key: flattened[key] for key in meta_keys}

        # Flatten the data into a DataFrame
        output_flattened = pd.json_normalize(data=output_dict)

        # Create a DataFrame with the specified columns, defaulting missing
        # values to 0.0
        output_flattened_filtered = pd.DataFrame(columns=columns)

        # Set all columns in output_flattened_filtered to 0.0 initially
        output_flattened_filtered = output_flattened_filtered.fillna(0.0)

        # Update only the matching columns from output_flattened
        for col in output_flattened.columns.intersection(columns):
            output_flattened_filtered[col] = output_flattened[col]

        # Ensure the order of columns matches the 'columns' list exactly
        output_flattened_filtered = output_flattened_filtered[columns]

        output_flattened_filtered = output_flattened_filtered.replace(
            [np.inf, -np.inf, np.nan], 0.0
        )

        # Convert the DataFrame to a dictionary format required for the JSON
        # payload
        json_data_payload = output_flattened_filtered.to_dict(orient="records")[0]

        return (json_data_payload, output_meta)
