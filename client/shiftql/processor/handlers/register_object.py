from .base import QueryObject


class RegisterObject(QueryObject):
    def __init__(self, register_object) -> None:
        self.meta_keys = ["size", "date_added"]
        super().__init__(register_object)

    def process(self):
        self.process_tables()
        self.process_info()
        self.process_dataobject()

    def process_tables(self):
        self.register_table = self.object["table"]

    def process_info(self):
        self.info = {"name": self.object["title"]}

    def process_dataobject(self):
        self.register_object = {}
        self.info_object = {}
        for each_column, each_value in zip(
            self.object["columns"], self.object["values"]
        ):
            if each_column["name"] in self.meta_keys:
                self.info[each_column["name"]] = each_value["name"]
            else:
                self.register_object[each_column["name"]] = each_value["name"]
        print(self.register_object)
