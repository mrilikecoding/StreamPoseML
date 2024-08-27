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


class ThirtyFrameJointsTransformer(SequenceTransformer):
    # joints.nose.x	joints.nose.y	joints.nose.z	joints.left_eye_inner.x	joints.left_eye_inner.y	...	joints.left_foot_index.z	joints.right_foot_index.x	joints.right_foot_index.y	joints.right_foot_index.z	joints.neck.x	joints.neck.y	joints.neck.z	joints.mid_hip.x	joints.mid_hip.y	joints.mid_hip.z

    @staticmethod
    def normalize_keypoints(group):
        """ TODO extract to util """
        # Choose a reference keypoint, for example, 'joints.nose.z'
        reference_keypoint = 'joints.nose.z'
        
        # Calculate the initial distance using the first frame of this group
        initial_distance = group[reference_keypoint].iloc[0]
        
        # Compute scale factors for each frame in the group
        scale_factors = initial_distance / group[reference_keypoint]
        
        # Apply scaling to x and y coordinates
        for col in group.columns:
            if col.endswith('.x') or col.endswith('.y'):
                group[col] = group[col] * scale_factors
        
        # Re-center the keypoints
        centre_x = (group[[col for col in group.columns if col.endswith('.x')]].min(axis=1) + 
                    group[[col for col in group.columns if col.endswith('.x')]].max(axis=1)) / 2
        centre_y = (group[[col for col in group.columns if col.endswith('.y')]].min(axis=1) + 
                    group[[col for col in group.columns if col.endswith('.y')]].max(axis=1)) / 2
        
        for col in group.columns:
            if col.endswith('.x'):
                group[col] = group[col] - centre_x
            elif col.endswith('.y'):
                group[col] = centre_y - group[col] 

        
        return group

    def transform(self, data: any, columns: list) -> any:
        joint_list = [
            "nose",
            "left_eye_inner",
            "left_eye",
            "left_eye_outer",
            "right_eye_inner",
            "right_eye",
            "right_eye_outer",
            "left_ear",
            "right_ear",
            "mouth_left",
            "mouth_right",
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
            "left_pinky",
            "right_pinky",
            "left_index",
            "right_index",
            "left_thumb",
            "right_thumb",
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_ankle",
            "left_heel",
            "right_heel",
            "left_foot_index",
            "right_foot_index",
            "neck",
            "mid_hip",
        ]
        df_data = []
        column_names = [f"joints.{joint_name}.{dimension}" for joint_name in joint_list for dimension in ["x", "y", "z"]]
        frames = data["frames"]
        frame_joint_positions = []
        dummy_columns = ['pid', 'video_index', 'is_step', 'weight_transfer_type', 'step_type']
        column_names = dummy_columns + column_names
        for frame in frames:
            joint_set = frame["joint_positions"]
            frame_joint_positions.append(joint_set)
        for joint_set in frame_joint_positions:
            frame_data = []
            for c in dummy_columns:
                if c == "weight_transfer_type":
                    frame_data.append(0.0)
                else:
                    frame_data.append(0)
            for joint_name in joint_list:
                for dimension in ["x", "y", "z"]:
                    value = joint_set[joint_name][dimension]
                    frame_data.append(value)
            df_data.append(frame_data)
        df = pd.DataFrame(df_data, columns=column_names)
        normalized_df = df.groupby(['pid', 'video_index']).apply(self.normalize_keypoints).reset_index(drop=True)
        return normalized_df, {}







