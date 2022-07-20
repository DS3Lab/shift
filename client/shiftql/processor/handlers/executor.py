import time
import timeit
from typing import List

from processor.handlers.http_request import base_query, simplify_reader_by_json

from common.classifier import Classifier
from common.query import QueryTypes
from loguru import logger
from processor.handlers.base import QueryObject
from processor.handlers.rank_object import RankObject
from processor.handlers.declare_object import DeclareObject
from processor.handlers.executors.finetune_exec import FinetuneExecutor
from processor.handlers.executors.purge_exec import PurgeExecutor
from processor.handlers.executors.rank_exec import RankExecutor
from processor.handlers.post_processing.set_operators import process_other
from processor.handlers.register_object import RegisterObject
from processor.handlers.select_object import SelectObject
from processor.parser import Parser
from rich.console import Console


class Executor:
    def __init__(self, debug) -> None:
        self.server_url = "http://127.0.0.1:8001"
        self.debug = debug
        self.parser = Parser(debug)
        self.scope_vars = {
            "readers": None,
            "models": None,
            "query_type": None,
            "response": None,
            "remaining_tasks": 0,
            "custom_vars": [],
        }
        self.console = Console()
        self.finetune_executor = FinetuneExecutor(self.server_url)
        self.purge_executor = PurgeExecutor(self.server_url)
        # here we will use localhost temporarily, eventually this will be configured by the user
        self.rank_executor = RankExecutor(self.server_url, "http://localhost:5050/")

    def _query(self, method, endpoint, data=None):
        return base_query(method, self.server_url + endpoint, data)

    def _query_reader_by_name(self, name):
        return self._query("POST", "/query_reader_by_name", data={"name": name})

    """
    process_stmt is the entry point for all different statements
    """

    def process_stmt(self, stmt: str):
        stmt = stmt.strip()
        self.stmt = stmt
        if stmt and not stmt.startswith("#"):
            """
            The hash sign # is used as an indicator of comment line.
            We only support single line comments.

            We also trim the query string such that the white spaces will not affect the results.
            """
            self.console.print("[IN]: {}".format(stmt))
            query_object = self.parser.parse(stmt)
            if self.debug:
                print(query_object)
            if query_object["type"] == "USE":
                self.handle_use(query_object)
                return True
            elif query_object["type"] == "SELECT":
                select_obj = SelectObject(
                    query_object, self.server_url, custom_vars=self.scope_vars
                )
                self.handle_query(select_obj)
                return True
            elif query_object["type"] == "REGISTER":
                query_object = RegisterObject(query_object)
                self.handle_register(query_object)
                return True
            elif query_object["type"] == "DECLARE":
                query_object = DeclareObject(query_object)
                self.handle_declare(query_object)
                return True
            elif query_object["type"] == "PRINT":
                self.handle_print(query_object)
                return True
            elif query_object["type"] == "EXPLAIN":
                self.handle_explain(query_object)
                return True
            elif query_object["type"] == "FINETUNE":
                self.handle_finetune(query_object)
            elif query_object["type"] == "PURGE":
                self.handle_purge(query_object)
            else:
                raise ValueError(
                    "Unsupported statement of the type {}".format(query_object["type"])
                )
        else:
            return False

    """
    Below are handlers for different types of statements
    """

    def handle_finetune(self, query_obj):
        self.finetune_executor.execute(query_obj, self.scope_vars)

    def handle_print(self, print_obj):
        if print_obj["item"] == "*":
            print(self.scope_vars)
        else:
            print(self.scope_vars[print_obj["item"]])
        self.scope_vars["query_type"] = QueryTypes.PRINT
        self.scope_vars["response"] = None

    def handle_purge(self, query_object):
        self.purge_executor.execute(query_object, self.scope_vars)

    def handle_use(self, use_obj):
        self.server_url = use_obj["hostname"]
        self.scope_vars["query_type"] = QueryTypes.USE

    def handle_declare(self, declare_obj: DeclareObject):
        inner_select = SelectObject(declare_obj.query, self.server_url, self.scope_vars)
        result = self.handle_query(inner_select)
        # a hack for experiments
        variable_name = declare_obj.variable_name
        if variable_name == 'pool':
            self.pool = result
        self.scope_vars[variable_name] = {
            "from": inner_select.select_table,
            "value": result.copy(),
        }
        self.scope_vars["custom_vars"].append(variable_name)
        self.scope_vars["response"] = None

    def handle_explain(self, query_obj: QueryObject):
        logger.info("Started Explain...")
        if 'select' in query_obj:
            inner_select = SelectObject(
                query_obj["select"], self.server_url, self.scope_vars
            )
            start = timeit.default_timer()
            self.handle_query(inner_select)
            stop = timeit.default_timer()
            self.scope_vars["query_type"] = QueryTypes.EXPLAIN
            self.scope_vars["response"] = {
                "stmt": self.stmt,
                "benchmark": inner_select.benchmark,
                "order": inner_select.order,
                "order_type": inner_select.order_type,
                "total_execution_time": stop - start,
                "remaining_tasks": self.scope_vars["remaining_tasks"],
                "models": self.scope_vars["models"],
                "custom_vars": self.scope_vars["custom_vars"],  # for backup only
            }
            if query_obj["output"]:
                if query_obj["output"] == "JSON":
                    import json

                    ts = time.time()
                    with open("experiments/{}.json".format(str(ts)), "x") as fp:
                        json.dump(self.scope_vars["response"], fp)
        elif 'rank' in query_obj:
            query_obj = RankObject(query_obj['rank'], self.server_url)
            self.rank_executor.execute(query_obj, self.scope_vars)

    def handle_query(self, select_obj: SelectObject):
        if select_obj.wait and select_obj.order_type == "metric":
            response = self.select_models(select_obj)
            while response[1] > 0:
                print(
                    "{} remaining tasks are still running, I am collecting...".format(
                        response[1] if response[1] <= 1000 else "some"
                    )
                )
                response = self.select_models(select_obj, dry_run=True)
                time.sleep(5)
            response = self.select_models(select_obj)
            self.scope_vars["remaining_tasks"] = response[1]
            self.scope_vars["models"] = response[0]
            self.scope_vars["response"] = (response[0], response[1])
            self.scope_vars["query_type"] = QueryTypes.QUERY_MODELS
            return response[0]

        elif select_obj.select_type == "models":
            if select_obj.order_type == "metric":
                response = self.select_models(select_obj)
                self.scope_vars["remaining_tasks"] = response[1]
                self.scope_vars["models"] = response[0]
                self.scope_vars["response"] = (response[0], response[1])
                self.scope_vars["query_type"] = QueryTypes.QUERY_MODELS
                return response[0]
            else:
                response = self.select_models(select_obj)
                self.scope_vars["models"] = response
                self.scope_vars["response"] = response
                self.scope_vars["query_type"] = QueryTypes.RESTRICT_MODELS
                return response

        elif select_obj.select_type == "readers":
            response = self.select_readers(select_obj)
            self.scope_vars["readers"] = response
            self.scope_vars["response"] = response
            if not select_obj.distance_mat:
                self.scope_vars["query_type"] = QueryTypes.SELECT_READERS
            else:
                self.scope_vars["query_type"] = QueryTypes.TASK2VEC_MATRIX
            return response

        elif select_obj.restricted:
            # if it is already restricted - we already have these objects, we just simply fill the values
            self.scope_vars["response"] = self.scope_vars[select_obj.select_table][
                "value"
            ]
            if self.scope_vars[select_obj.select_table]["from"] in [
                "text_models",
                "image_models",
            ]:
                self.scope_vars["query_type"] = QueryTypes.SELECT_MODELS
            elif self.scope_vars[select_obj.select_table]["from"] in ["readers"]:
                self.scope_vars["query_type"] = QueryTypes.SELECT_READERS
            return self.scope_vars[select_obj.select_table]

    def handle_register(self, register_obj: RegisterObject):
        register_query = {
            register_obj.register_table: register_obj.register_object,
            "info": register_obj.info,
        }
        print(register_query)
        if register_obj.register_table == "dataset":
            self.register_dataset(register_query)

    """
    Functions below are for specific goals
    """

    def restrict_models_pool(self, select_obj: SelectObject):
        table = select_obj.select_table
        responses = {"models": [], "info": {"num_params": [], "up_acc": []}}
        if select_obj.where_set_operation is None:
            responses = self._query(
                "POST", "/" + table, data=select_obj.where_payloads[0]
            )
        elif select_obj.where_set_operation == "AND":
            responses = self._query("POST", "/" + table, data=select_obj.where_payloads)
        elif select_obj.where_set_operation == "OR":
            for each in select_obj.where_payloads:
                response = self._query("POST", "/" + table, data=each)
                responses["models"] += response["models"]
                if "info" in response:
                    for key in responses["info"]:
                        responses["info"][key] += response["info"][key]
        else:
            raise ValueError(
                "Unsupported set operation {}".format(select_obj.where_set_operation)
            )
        return responses

    def select_readers(self, select_obj: SelectObject):
        responses = []
        if select_obj.distance_mat:
            return self.get_task2vec_distance_matrix(select_obj)
        for where_payload in select_obj.where_payloads:
            if not where_payload:
                response = self._query("GET", "/readers")
            else:
                response = self._query("POST", "/query_reader", data=where_payload)
            responses += response
        return responses

    def get_task2vec_distance_matrix(self, select_obj: SelectObject):
        if select_obj.distance_mat:
            readers = self.scope_vars[select_obj.select_table]["value"]
            readers = [reader["json"] for reader in readers]
            probe = select_obj.where_payloads[0]["probe"]
            probe = self.scope_vars[probe]["value"]["models"][0]

            response = self._query(
                "POST", "/task2vec", data={"probe": probe, "readers": readers}
            )
            self.scope_vars["readers"] = readers
            self.scope_vars["response"] = response
            return {
                "readers": self.reorganize_task2vec_results(
                    response["distances"], readers
                ),
                "remaining_tasks": response["num_remaining_tasks"],
            }

    def select_models(self, select_obj: SelectObject, dry_run=False):
        # if no order is needed:
        if not select_obj.order_type:
            if select_obj.restricted:
                return self.scope_vars[select_obj.select_table]["value"]
            response = self.restrict_models_pool(select_obj)

            if select_obj.limit:
                response["models"] = response["models"][0 : select_obj.limit]
            if select_obj.postprocessing_payloads:
                results = []
                for each in response["models"]:
                    for each_filter in select_obj.postprocessing_payloads:
                        filter_key = list(each_filter.keys())[0]
                        if each[filter_key] == each_filter[filter_key]:
                            results.append(each)
                response["models"] = results
            return response

        if select_obj.order_type == "column":
            if select_obj.restricted:
                response = self.scope_vars[select_obj.select_table]["value"]
            else:
                response = self.restrict_models_pool(select_obj)

            # here add the information fields into the response
            for idx, each in enumerate(response["models"]):
                if "info" in response:
                    each["num_params"] = response["info"]["num_params"][idx]
                    each["up_acc"] = response["info"]["up_acc"][idx]
            if "info" in response:
                del response["info"]
            response["models"].sort(
                key=lambda x: x[select_obj.order], reverse=select_obj.order_reverse
            )
            if select_obj.limit:
                response["models"] = response["models"][0 : select_obj.limit]
            return response

        elif select_obj.order_type == "metric":
            if select_obj.restricted and self.scope_vars[select_obj.select_table] and select_obj.other_than is None:
                restricted_model = self.scope_vars[select_obj.select_table]["value"]["models"]
                for each in restricted_model:
                    if "num_params" in each:
                        del each["num_params"]
                    if "up_acc" in each:
                        del each["up_acc"]
                known_results, remaining_tasks = self.query_models_with_metrics(
                    select_obj,
                    self.scope_vars[select_obj.select_table]["value"]["models"],
                    dry_run=dry_run,
                )
            else:
                if select_obj.other_than is not None:
                    # here we have specified a model to be excluded
                    remaining_candidates = process_other(
                        self.pool['models'],
                        self.scope_vars[select_obj.other_than],
                    )
                    for each in remaining_candidates:
                        if "num_params" in each:
                            del each["num_params"]
                        if "up_acc" in each:
                            del each["up_acc"]
                    known_results, remaining_tasks = self.query_models_with_metrics(
                        select_obj, remaining_candidates, dry_run=dry_run
                    )
                else:
                    known_results, remaining_tasks = self.query_models_with_metrics(
                        select_obj, dry_run=dry_run
                    )
            known_results = self.split_results_by_classifier(select_obj, known_results)

            for each_classifier in known_results:
                known_results[each_classifier] = self.simplify_benchmark_results(
                    select_obj, known_results[each_classifier], each_classifier
                )
                known_results[each_classifier].sort(
                    key=lambda x: x[select_obj.order], reverse=select_obj.order_reverse
                )
                if select_obj.limit:
                    known_results[each_classifier] = known_results[each_classifier][
                        0 : select_obj.limit
                    ]
            return (known_results, remaining_tasks)
        return None, None

    def query_models_with_metrics(
        self, select_obj: SelectObject, models=None, dry_run=False
    ):
        if not models:
            models = self.restrict_models_pool(select_obj)
            models = models["models"]
        classifiers = select_obj.classifier
        if select_obj.benchmark:
            query_payload = {
                "train": select_obj.train,
                "test": select_obj.test,
                "models": models,
                "classifiers": classifiers,
                "benchmark": True,
                "dry": dry_run,
                "chunk_size": select_obj.chunk
                if select_obj.chunk is not None
                else None,
                "budget": select_obj.budget if select_obj.budget is not None else None,
            }
        else:
            query_payload = {
                "train": select_obj.train,
                "test": select_obj.test,
                "models": models,
                "classifiers": classifiers,
                "limit": select_obj.limit if select_obj.limit else 0,
                "dry": dry_run,
                "chunk_size": select_obj.chunk
                if select_obj.chunk is not None
                else None,
                "budget": select_obj.budget if select_obj.budget is not None else None,
            }
        # check if there is any none value
        response = self._query("POST", "/query", data=query_payload)
        return response["known_results"], response["num_remaining_tasks"]

    def register_dataset(self, register_obj: dict):
        try:
            response = self._query("POST", "/register_reader", data=register_obj)
            print("Success")
        except Exception as e:
            # TODO: Make a new type of exception to capture.
            # Make hints on updates/new insertions/etc.
            print("Error")
            raise ValueError(e)

    """
    Utilities
    """

    def split_results_by_classifier(self, select_obj: SelectObject, known_results):
        reorganized_known_results = {}
        for each in select_obj.classifier:
            if "parameters" in each:
                each = Classifier(name=each["name"], parameters=each["parameters"])
            else:
                each = Classifier(name=each["name"])
            reorganized_known_results[each.value] = [
                res for res in known_results if res["classifier"] == each.value
            ]
        return reorganized_known_results

    def reorganize_task2vec_results(self, distances, readers):
        """
        reorganize the results into the following format:
        {
            "reader_1": {
                "reader_1": distance,
                "reader_2": distance
            }
        }
        """
        import json

        readers = [
            simplify_reader_by_json(self.server_url, json.dumps(reader))["name"]
            for reader in readers
        ]
        results = {}

        # TODO: hints on non-complete results - https://trello.com/c/wpO4mpSt/15-hints-on-non-complete-results
        for idx_from, each_from in enumerate(readers):
            results[each_from] = {}
            for idx_to, each_to in enumerate(readers):
                results[each_from][each_to] = distances[idx_from][idx_to]
        return results

    def simplify_benchmark_results(
        self, select_obj: SelectObject, known_results: List, each_classifier
    ):
        if select_obj.benchmark and select_obj.order_func:
            reduced_known_results = []
            readers_per_model = {}
            models = [str(each["json_model"]) for each in known_results]
            models = list(set(models))
            for each in models:
                readers_per_model[each] = []
            for each in known_results:
                readers_per_model[str(each["json_model"])].append(each)
            for each in models:
                errors_per_model = []
                for each_reader in readers_per_model[str(each)]:
                    errors_per_model.append(each_reader["err"])
                weighted_error = select_obj.order_func(errors_per_model)
                reduced_known_results.append(
                    {
                        "json_model": each,
                        "err": weighted_error,
                        "classifier": each_classifier,
                        "func": select_obj.order_func_name,
                    }
                )
            return reduced_known_results
        else:
            return known_results
