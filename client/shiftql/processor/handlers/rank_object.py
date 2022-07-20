from common.query import ReservedKeywords

from .base import QueryObject
from .http_request import resolve_reader_by_name


def get_order_function(order_func: str):
    if order_func.lower() == "max":
        return max
    elif order_func.lower() == "min":
        return min
    elif order_func.lower() == "sum":
        return sum
    elif order_func.lower() == "avg":
        return lambda x: sum(x) / len(x)
    else:
        raise Exception("Order function not supported: {}".format(order_func))


class RankObject(QueryObject):
    def __init__(self, select_query_object, server_url, custom_vars=[]):
        self.server_url = server_url
        self.custom_vars = custom_vars
        self.where_set_operation = None
        self.where_payloads = None
        super().__init__(select_query_object)

    def __str__(self) -> str:
        return str({
            "custom_vars": self.custom_vars,
            "restricted": self.restricted,
            "where": self.where_payloads,
            "by": self.by,
            "train": self.train,
            "test": self.test,
        })

    def process(self):
        self.process_columns()
        self.process_tables()
        if self.object["where"] is not None:
            self.process_where_clause()
        self.process_order_by()
        self.process_on()
        self.process_limit()
        self.process_wait()
        self.process_by()

    def process_by(self):
        print(self.object)
        self.by = self.object['rank_by']['by']

    def process_columns(self):
        self.columns = []
        self.distance_mat = False
        for each in self.object["column"]:
            self.columns.append(each["name"])
            if each["name"] == "vec":
                self.distance_mat = True

    def process_tables(self):
        select_table = self.object["table"]
        # if it is a reserved keyword --> it has not been restricted.
        # here the restricted is from "restrict model pool".
        if ReservedKeywords.contains(select_table):
            self.restricted = False
        else:
            self.restricted = True
        select_type = None
        if select_table in ["image_models", "text_models"]:
            select_type = "models"
        elif select_table in ["readers"]:
            select_type = "readers"
        elif select_table in self.custom_vars:
            if self.custom_vars[select_table]["from"] in [
                "image_models",
                "text_models",
            ]:
                select_type = "models"
            if self.custom_vars[select_table]["from"] in ["readers"]:
                select_type = "readers"
        else:
            raise ValueError("Not supported table {}".format(select_table))
        self.select_type, self.select_table = select_type, select_table

    def process_where_clause(self):
        _where_clause = self.object["where"]
        set_operation = set(_where_clause[i] for i in range(1, len(_where_clause), 2))
        if len(set_operation) > 1:
            raise ValueError(
                "Cannot use 'AND', 'OR' together at the same time, it is not supported yet."
            )
        if len(set_operation) == 1:
            self.where_set_operation = list(set_operation)[0].upper()
        else:
            self.where_set_operation = None
        self.where_payloads = []
        self.postprocessing_payloads = []
        # it is used to query the database with certain conditions.
        # the keywords in a where clause forms a finite sets. We use a list to store the keywords and check if it is present.
        # In future, we may automatically fetch the format from the server, so that users' could somehow extend the columns.
        keywords = ["date_added", "num_params", "source", "up_acc", "finetuned"]
        if self.select_table == "image_models":
            keywords.append("image_size")
        if self.select_table == "text_models":
            keywords.append("token_length")
        if self.select_type == "readers":
            keywords.append("name")
            keywords.append("probe")
        self.column_keywords = keywords
        # then for each keyword in the list, we check if it is present in the where clause.
        # we form a query payload directly from the where clause.
        # TODO: This part can be re-written...
        where_payload = {}
        for i, each in enumerate(_where_clause):
            if i % 2 == 0:
                if each == "FINETUNED":
                    each = {"name": "finetuned"}
                    where_payload["finetuned"] = True
                elif each == "NOT FINETUNED":
                    each = {"name": "finetuned"}
                    where_payload["finetuned"] = False
                elif each["compare"] == "=":
                    where_payload[each["name"]] = each["value"]
                elif each["compare"] == ">":
                    if not each["name"] in where_payload:
                        where_payload[each["name"]] = {}
                    where_payload[each["name"]]["min"] = each["value"]
                elif each["compare"] == "<":
                    if not each["name"] in where_payload:
                        where_payload[each["name"]] = {}
                    where_payload[each["name"]]["max"] = each["value"]
                else:
                    raise ValueError(
                        "Unsupported compare operator: {}".format(each["compare"])
                    )

                if each["name"] in self.column_keywords:
                    self.where_payloads.append(where_payload)
                else:
                    self.postprocessing_payloads.append(where_payload)
                where_payload = {}

        if self.where_set_operation == "AND":
            self.where_payloads = self.merge_where_payload()

    def merge_where_payload(self):
        """
        merged_payloads is a struct with the following format:
        {
            "name": {...payload...}
        }
        """
        merged_payloads = {}
        for idx, each in enumerate(self.where_payloads):
            kw = list(each.keys())[0]
            if kw not in merged_payloads:
                # if it not seen before, simply append it
                merged_payloads[kw] = each[kw]
            else:
                # if it already in the merged_payloads, then merge them
                merged_payloads[kw] = {**merged_payloads[kw], **each[kw]}
        return merged_payloads

    def process_order_by(self):
        if self.object["order"]:
            order = self.object["order"]
            # There are two cases of order by:
            # 1. if the order is specified and presented in the columns keywords, we simply order the models.
            # 2. if the order is specified and it is related to a metric, e.g. err, then we query the known_results table and order the results.
            if order["name"] in ["err", "acc"]:
                needReverse = False if order["type"] == "ASC" else True
                self.order, self.order_reverse, self.order_type = (
                    order["name"],
                    needReverse,
                    "metric",
                )
            else:
                needReverse = False if order["type"] == "ASC" else True
                self.order, self.order_reverse, self.order_type = (
                    order["name"],
                    needReverse,
                    "column",
                )
            if order["function"] is not None:
                self.order_func_name = order["function"]
                self.order_func = get_order_function(order["function"])
        else:
            self.order, self.order_reverse, self.order_type = None, None, None

    def process_on(self):
        self.train = []
        self.test = []
        self.task_agnostic = False
        if self.object["tested_on"] and not self.object["trained_on"]:
            self.task_agnostic = True
        if self.object["tested_on"]:
            if self.object["tested_on"]["task_type"] == "BENCHMARK":
                self.benchmark = True
            else:
                self.benchmark = False
            tested_datasets = self.object["tested_on"]["datasets"]
            if (
                len(tested_datasets) == 1
                and tested_datasets[0]["name"] in self.custom_vars
            ):
                self.test = self.resolve_prefetched_reader(tested_datasets[0]["name"])
            else:
                for each in tested_datasets:
                    reader = resolve_reader_by_name(each["name"], self.server_url)
                    self.test.append(
                        {
                            "reader": reader,
                        }
                    )
        if self.object["trained_on"]:
            trained_datasets = self.object["trained_on"]["datasets"]
            if (
                len(trained_datasets) == 1
                and trained_datasets[0]["name"] in self.custom_vars
            ):
                self.train = self.resolve_prefetched_reader(trained_datasets[0]["name"])
            else:
                for each in trained_datasets:
                    reader = resolve_reader_by_name(each["name"], self.server_url)
                    self.train.append(
                        {
                            "reader": reader,
                        }
                    )

    def process_limit(self):
        self.limit = self.object["limit"]

    def process_wait(self):
        self.wait = self.object["wait"]

    def resolve_prefetched_reader(self, name):
        return [{"reader": each["json"]} for each in self.custom_vars[name]["value"]]
