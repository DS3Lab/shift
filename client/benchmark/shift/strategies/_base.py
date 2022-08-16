from shift.io.api.client import ShiftAPI

class BaseSearchStrategy():
    def __init__(self) -> None:
        self.shift_api = ShiftAPI()

    def search(self):
        """
        Search for the possibly best models based on the certain strategy and its configuration.
        Returns:
            - List of models.
            - Cost of time in order to find these models
        """
        raise NotImplementedError

    def report(self):
        raise NotImplementedError

    def set_search_config(self, config_string):
        self._parse_config(config_string)

    def _parse_config(self, yaml_config_string):
        self.config = yaml_config_string
        self.target = self.config['target']
        if 'meta' in self.config:
            self.meta = self.config['meta']
        self._find_all_candidate_models()

    def _find_all_candidate_models(self):
        self.candidate_models = self.shift_api.get_known_models(self.target)
        return self.candidate_models
