from processor.handlers.executor import base_query


class FinetuneExecutor:
    def __init__(self, endpoint) -> None:
        self.endpoint = endpoint

    def execute(self, finetune_obj, scope_vars) -> None:
        self.finetune_obj = finetune_obj
        if self.finetune_obj["model"] in scope_vars["custom_vars"]:
            self.finetune_obj["model"] = scope_vars[self.finetune_obj["model"]][
                "value"
            ]["models"]
        if self.finetune_obj["reader"] in scope_vars["custom_vars"]:
            self.finetune_obj["reader"] = scope_vars[self.finetune_obj["reader"]][
                "value"
            ]

        readers = [reader["json"] for reader in self.finetune_obj["reader"]]
        for each_model in self.finetune_obj["model"]:
            if "num_params" in each_model:
                del each_model["num_params"]
            if "up_acc" in each_model:
                del each_model["up_acc"]
            query_payload = {
                "model": each_model,
                "readers": readers,
                "lr": 0.01,
                "epochs": 50,
            }
            base_query(
                method="POST", endpoint=self.endpoint + "/finetune", data=query_payload
            )
