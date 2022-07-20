from processor.handlers.http_request import base_query, simplify_reader_by_json

class RankExecutor:
    def __init__(self, shift_endpoint, ext_endpoint) -> None:
        self.ext_endpoint = ext_endpoint
        self.shift_endpoint = shift_endpoint

    def _query(self, method, endpoint, data=None):
        return base_query(method, self.shift_endpoint + endpoint, data)

    def _query_reader_by_name(self, name):
        return self._query("POST", "/query_reader_by_name", data={"name": name})

    def _query_ext(self, method, endpoint, data=None):
        return base_query(method, self.ext_endpoint + endpoint, data=data)

    def execute(self, rank_obj, scope_vars):
        print(rank_obj)
        response = self.restrict_models_pool(rank_obj)
        candidate_models = response["models"]
        results = self._query_ext("POST", f"ext/{rank_obj.by}", data={
            "models": candidate_models,
            "readers": {
                "train": rank_obj.train,
                "test": rank_obj.test
            },
        })
        return results

    def restrict_models_pool(self, rank_obj):
        table = rank_obj.select_table
        responses = {"models": [], "info": {"num_params": [], "up_acc": []}}
        if rank_obj.where_set_operation is None:
            responses = self._query(
                "POST", "/" + table, data=rank_obj.where_payloads[0]
            )
        elif rank_obj.where_set_operation == "AND":
            responses = self._query("POST", "/" + table, data=rank_obj.where_payloads)
        elif rank_obj.where_set_operation == "OR":
            for each in rank_obj.where_payloads:
                response = self._query("POST", "/" + table, data=each)
                responses["models"] += response["models"]
                if "info" in response:
                    for key in responses["info"]:
                        responses["info"][key] += response["info"][key]
        else:
            raise ValueError(
                "Unsupported set operation {}".format(rank_obj.where_set_operation)
            )
        return responses