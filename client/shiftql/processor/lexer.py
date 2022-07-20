"""
This file describes the tokens and their regex extractor
"""
from processor.exceptions import LexerException

RESERVED_KEYWORDS = {
    "print": "PRINT",
    "use": "USE",
    "select": "SELECT",
    "finetune": "FINETUNE",
    "declare": "DECLARE",
    "where": "WHERE",
    "between": "BETWEEN",
    "from": "FROM",
    "order": "ORDER",
    "desc": "DESC",
    "other": "OTHER",
    "than": "THAN",
    "chunk": "CHUNK",
    "budget": "BUDGET",
    "asc": "ASC",
    "explain": "EXPLAIN",
    "limit": "LIMIT",
    "as": "AS",
    "by": "BY",
    "and": "AND",
    "with": "WITH",
    "is": "IS",
    "or": "OR",
    "on": "ON",
    "finetuned": "FINETUNED",
    "not": "NOT",
    "null": "NULL",
    "sum": "SUM",
    "avg": "AVG",
    "min": "MIN",
    "max": "MAX",
    "benchmark": "BENCHMARK",
    "task": "TASK",
    "distinct": "DISTINCT",
    "register": "REGISTER",
    "values": "VALUES",
    "tested": "TESTED",
    "trained": "TRAINED",
    "classified": "CLASSIFIED",
    "rank":"RANK",
    "ranked": "RANKED",
    "change": "CHANGE",
    "to": "TO",
    "purge": "PURGE",
}

tokens = (
    "COMPARISON",
    "END",
    "STRING",
    "NUMBER",
    "QSTRING",
    "COMMA",
) + tuple(set(RESERVED_KEYWORDS.values()))

literals = "(){}@%.*[]:-^"
t_COMPARISON = r"<>|!=|>=|<=|=|>|<"
t_END = r";"
t_COMMA = r","
t_ignore = " \t\n"


def t_STRING(t):
    r"[a-zA-Z][_a-zA-Z0-9]*"
    t.type = RESERVED_KEYWORDS.get(t.value.lower(), "STRING")
    if t.type != "STRING":
        t.value = t.value.upper()
    return t


def t_QSTRING(t):
    r"('[^']*')|(\"[^\"]*\")|(`[^`]*`)"
    t.value = t.value[1:-1]
    return t


def t_NUMBER(t):
    r"\d+(\.\d+)?"
    try:
        t.value = int(t.value)
    except ValueError:
        t.value = float(t.value)
    return t


def t_error(t):
    raise LexerException(
        "Illegal character '%s' at line %s pos %s" % (t.value[0], t.lineno, t.lexpos)
    )
