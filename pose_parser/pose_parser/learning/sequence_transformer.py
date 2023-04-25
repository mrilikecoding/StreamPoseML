from abc import ABC, abstractmethod


class SequenceTransformer(ABC):
    @abstractmethod
    def transform(self, data: any) -> any:
        pass


# TODO create concrete classes for different schemes
