import time

from processor.handlers.executor import base_query


class PurgeExecutor:
    def __init__(self, endpoint) -> None:
        self.endpoint = endpoint

    def execute(self, purge_obj, scope_vars) -> None:
        base_query(
            method="GET",
            endpoint=self.endpoint + "/purge",
        )
        print("done, waiting 5 secs")
        time.sleep(5)
