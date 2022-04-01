from processor.lexer import RESERVED_KEYWORDS
from prompt_toolkit import prompt
from prompt_toolkit.completion import WordCompleter

candidates = list(RESERVED_KEYWORDS.keys()) + [
    "text_models",
    "image_models",
    "readers",
    "datasets",
]

shiftql_completer = WordCompleter(candidates)
