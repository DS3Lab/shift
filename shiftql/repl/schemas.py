from enum import Enum

from common.query import ReturnedType
from tabulate import tabulate


class ReturnedList:
    def __init__(self, return_type, raw_data, query_object) -> None:
        if return_type not in ReturnedType:
            print("Unrecognized returned type: {}".format(return_type))
            print("Formatting may not work")
        self.return_type = return_type
        self.raw_data = raw_data
        self.query_object = query_object

    def reformat_readers(self):
        results = []
        for each in self.raw_data:
            result = {}
            for attr in each:
                if attr == "json":
                    for each_key in each[attr]:
                        result[each_key] = each[attr][each_key]
                else:
                    result[attr] = each[attr]
            results.append(result)
        return results

    def reformat_models(self):
        results = []
        for each in self.raw_data:
            result = {}
            for attr in each:
                if attr == "models":
                    for each_key in each[attr]:
                        result[each_key] = each[attr][each_key]
                else:
                    result[attr] = each[attr]
            results.append(result)
        return results

    def print(self):
        if self.return_type == ReturnedType.models:
            formatted_result = self.reformat_models()
        elif self.return_type == ReturnedType.readers:
            formatted_result = self.reformat_readers()
        if self.return_type not in ReturnedType:
            print(self.raw_data)
        else:
            print(tabulate(formatted_result, headers="keys", tablefmt="github"))
