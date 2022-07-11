from common.query import ReservedKeywords

from .base import QueryObject


class DeclareObject(QueryObject):
    def __init__(self, declare_query_object) -> None:
        super().__init__(declare_query_object)

    def process(self):
        self.query = self.object["select"]
        if ReservedKeywords.contains(self.object["variable"]):
            raise ValueError(
                'Cannot use reserved keyword "{}" as variable name.'.format(
                    self.object["variable"]
                )
            )
        self.variable_name = self.object["variable"]
