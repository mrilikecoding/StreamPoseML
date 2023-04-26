from abc import ABC, abstractmethod

import pandas as pd


class SequenceTransformer(ABC):
    @abstractmethod
    def transform(self, data: any) -> any:
        pass


# TODO create concrete classes for different schemes


class TenFrameFlatColumnAngleTransformer(SequenceTransformer):
    def transform(self, data: any) -> any:
        # value_map = {
        #     "weight_transfer_type": {
        #         "Failure Weight Transfer": 0,
        #         "Successful Weight Transfer": 1,
        #     },
        #     "step_type": {
        #         "Left Step": 0,
        #         "Right Step": 1,
        #     },
        # }
        # drop_list = ["video_id"]
        # column_whitelist = []
        return []
