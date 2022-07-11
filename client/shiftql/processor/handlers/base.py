from abc import ABC, abstractmethod


class QueryObject(ABC):
    def __init__(self, parsed_object: dict) -> None:
        self.object = parsed_object
        # uncomment the line below for debugging
        # print(">>> parsed into: {}".format(self.object))
        self.process()

    @abstractmethod
    def process(self):
        raise NotImplementedError
