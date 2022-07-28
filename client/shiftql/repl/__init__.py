from colorama import Fore, init
from processor.handlers.executor import Executor
from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.styles import Style
from repl.completer import shiftql_completer
from repl.formatter import Printable

rprompt = Style.from_dict(
    {
        "rprompt": "bg:#ff0066 #ffffff",
    }
)

def server_toolbar(server_addr: str):
    return HTML("Connected to {}".format(server_addr))

def get_rprompt(text):
    return text

def process_stmts(stmts, debug):
    executor = Executor(debug)
    for each in stmts:
        if executor.process_stmt(each):
            response_type = executor.scope_vars["query_type"]
            response = executor.scope_vars["response"]
            if response_type is not None and response is not None:
                Printable(executor.scope_vars).print()

class REPL:
    def __init__(self, debug) -> None:
        self.history = InMemoryHistory()
        self.failure = lambda x: f"{Fore.RED}{x}"
        self.session = PromptSession(history=self.history)
        self.executor = Executor(debug)
        init(autoreset=True)

    def execute(self, stmt: str) -> None:
        if self.executor.process_stmt(stmt):
            response_type = self.executor.scope_vars["query_type"]
            response = self.executor.scope_vars["response"]
            if response_type is not None and response is not None:
                Printable(
                    response, response_type, scope=self.executor.scope_vars
                ).print()

    def run(self):
        print()
        print("Welcome to ShiftQL")
        print("The Search Engine for Machine Learning")
        print("ctrl-c to quit")
        print()
        try:
            while True:
                try:
                    _in = self.session.prompt(
                        ">>> ",
                        auto_suggest=AutoSuggestFromHistory(),
                        completer=shiftql_completer,
                        complete_while_typing=False,
                        bottom_toolbar=server_toolbar(self.executor.server_url),
                    )
                    self.execute(_in)
                except Exception as e:
                    print(self.failure("Error: {}".format(e)))
        except KeyboardInterrupt as e:
            print("Bye!")
