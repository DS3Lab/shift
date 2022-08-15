from rich.console import Console
from rich.table import Table

class Formatter(object):
    def __init__(self):
        self.console = Console()

    def print_tables(self, items, title):
        keys = set()
        if len(items) > 0:
            table = Table(title=title)
            for item in items:
                for key in item.keys():
                    keys.add(key)
            for key in keys:
                table.add_column(key, justify="left")
            for item in items:
                for key in keys:
                    if key not in item:
                        item[key] = ""
                table.add_row(*[str(item[key]) for key in keys])
            self.console.print(table)
        else:
            self.console.print("No Results Found...")
