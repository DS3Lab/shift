from common.query import QueryTypes
from tabulate import tabulate

from .format._base import Formatter
from .notification import send_explain


class Printable(object):
    def __init__(self, scope=None):
        self.scope = scope
        self.response = scope["response"]
        self.response_type = scope["query_type"]
        self.formatter = Formatter()

    def print(self):
        if self.response_type == QueryTypes.RESTRICT_MODELS:
            self.print_models()
        elif self.response_type == QueryTypes.QUERY_MODELS:
            self.print_query_results(self.response[0], self.response[1])
        elif self.response_type == QueryTypes.SELECT_READERS:
            self.print_readers()
        elif self.response_type == QueryTypes.EXPLAIN:
            self.response["models"] = self.scope["models"]
            self.print_query_results(
                self.response["models"], self.scope["remaining_tasks"]
            )
            self.print_explain()
        elif self.response_type == QueryTypes.TASK2VEC_MATRIX:
            self.print_task2vec_matrix()
        else:
            raise ValueError("Invalid response type")

    def print_models(self):
        self.formatter.print_tables(self.response["models"], title="Models")

    def print_readers(self):
        self.formatter.print_tables(self.response, title="Readers")

    def print_explain(self):
        print("-" * 15 + " Query Plan " + "-" * 15)
        print(
            "[Select] Order {}: {}. Benchmark: {}".format(
                self.response["order_type"],
                self.response["order"],
                self.response["benchmark"],
            )
        )
        print("-" * 42)
        print(
            "Query finished in {:.2f} seconds.".format(
                self.response["total_execution_time"]
            )
        )
        print("{} remaining task(s) left.".format(self.response["remaining_tasks"]))
        # send_explain(self.response["stmt"], self.response["total_execution_time"])

    def print_task2vec_matrix(self):
        num_remaining_tasks = self.response["remaining_tasks"]
        readers = self.response["readers"]
        if num_remaining_tasks == 0:
            print("\N{grinning face} Luckily, all results are known, as shown below:")
        else:
            print(
                "\N{confused face} Unfortunately, not all results are known ({} remaining). The known results are shown as below:".format(
                    num_remaining_tasks
                )
            )
        headers = list(readers.keys())
        """
        Reformat
        """
        print_data = []
        for each in readers:
            distances = [each for each in list(readers[each].values())]
            distances = [each] + distances
            print_data.append(distances)
        print(tabulate(print_data, headers=[" "] + headers, tablefmt="github"))

    def print_query_results(self, known_results, num_remaining_tasks):
        if num_remaining_tasks == 0:
            print("\N{hugging face} Luckily, all results are known, as shown below:")
        else:
            print(
                "\N{confused face} Unfortunately, not all results are known ({} remaining). The known results are shown as below:".format(
                    num_remaining_tasks
                )
            )
        print(known_results)
        for each in known_results:
            json_models = []
            errs = []
            starts = []
            stops = []
            classifiers = []
            has_start = False
            if len(known_results[each]) > 0:
                for res in known_results[each]:
                    json_models.append(res["json_model"])
                    errs.append(res["err"])
                    if (
                        "start" in res
                        and res["start"] is not None
                        and res["start"] != res["stop"]
                    ):
                        starts.append(res["start"])
                        stops.append(res["stop"])
                        has_start = True
                    classifiers.append(each)
                err_keyword = (
                    "{}(err)".format(known_results[each][0]["func"])
                    if "func" in known_results[each][0]
                    else "err"
                )
                print("For classifier: {}".format(each))
                if not has_start:
                    results = [
                        {"model": model, err_keyword: errs[idx], "classifier": each}
                        for idx, model in enumerate(json_models)
                    ]
                if has_start:
                    print(json_models)
                    results = [
                        {
                            "model": model,
                            err_keyword: errs[idx],
                            "classifier": each,
                            "start": starts[idx],
                            "stop": stops[idx],
                        }
                        for idx, model in enumerate(json_models)
                    ]
                self.formatter.print_tables(results, title="Models")
        if num_remaining_tasks != 0:
            print(
                "{} jobs are running and being evaluated, check it back later.".format(
                    num_remaining_tasks
                )
            )
