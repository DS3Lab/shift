from processor.exceptions import GrammarException

def p_expression(p):
    """expression : dml END"""
    p[0] = p[1]


def p_dml(p):
    """dml : select
    | use
    | register
    | declare
    | print
    | explain
    | finetune
    | purge
    | rank
    """
    p[0] = p[1]

def p_rank(p):
    """rank : RANK columns FROM table where order_by trained_on tested_on rank_by limit wait"""
    p[0] = {
        "type": p[1],
        "column": p[2],
        "table": p[4],
        "where": p[5],
        "order": p[6],
        "trained_on": p[7],
        "tested_on": p[8],
        "rank_by": p[9],
        "limit": p[10],
        "wait": p[11],
    }

def p_rank_by(p):
    """rank_by : BY string
    """
    p[0] = {"by": p[2]}

def p_print(p):
    """print : PRINT string"""
    p[0] = {"type": p[1], "item": p[2]}

def p_purge(p):
    """purge : PURGE"""
    p[0] = {"type": p[1]}

def p_declare(p):
    """declare : DECLARE string AS select"""
    p[0] = {"type": p[1], "variable": p[2], "select": p[4]}

def p_explain(p):
    """explain : EXPLAIN output select
    | EXPLAIN output rank
    """
    if p[3]['type'] == 'RANK':
        p[0] = {"type": p[1], "output": p[2], "rank": p[3]}
    else:
        p[0] = {"type": p[1], "output": p[2], "select": p[3]}
    
def p_output(p):
    """output : STRING
    | empty
    """
    if p[1] is None:
        p[0] = None
    elif p[1].upper() == "JSON":
        p[0] = "JSON"
    else:
        p[0] = None

def p_register(p):
    """register : REGISTER table title columns VALUES columns"""
    p[0] = {"type": p[1], "table": p[2], "title": p[3], "columns": p[4], "values": p[6]}

def p_title(p):
    """title : string"""
    p[0] = p[1]

def p_use(p):
    """use : USE string"""
    if len(p) == 3:
        p[0] = {"type": "USE", "hostname": p[2]}
    else:
        p[0] = {"type": "USE", "hostname": "127.0.0.1:8001"}

def p_string(p):
    """string : STRING
    | QSTRING
    | "*"
    """
    p[0] = p[1]

def p_finetune(p):
    """finetune : FINETUNE ft_model WITH ft_data wait"""
    p[0] = {
        "type": p[1],
        "model": p[2],
        "reader": p[4],
        "wait": p[5],
    }

def p_ft_model(p):
    """ft_model : string"""
    p[0] = p[1]

def p_ft_data(p):
    """ft_data : string"""
    p[0] = p[1]

def p_select(p):
    """select : SELECT columns FROM table where order_by trained_on tested_on classified_by limit other chunk budget wait"""
    p[0] = {
        "type": p[1],
        "column": p[2],
        "table": p[4],
        "where": p[5],
        "order": p[6],
        "trained_on": p[7],
        "tested_on": p[8],
        "classified_by": p[9],
        "limit": p[10],
        "other": p[11],
        "chunk": p[12],
        "budget": p[13],
        "wait": p[14],
    }


def p_chunk(p):
    """chunk : CHUNK NUMBER
    | empty
    """
    if len(p) > 2:
        p[0] = p[2]


def p_budget(p):
    """budget : BUDGET NUMBER
    | empty
    """
    if len(p) > 2:
        p[0] = p[2]


def p_wait(p):
    """wait : string
    | empty
    """
    p[0] = True if p[1] == "wait" else False


def p_table(p):
    """table : string"""
    p[0] = p[1]


def p_where(p):
    """where : WHERE conditions
    | string
    | empty
    """
    p[0] = []
    if len(p) > 2:
        p[0] = p[2]
    else:
        p[0] = p[1]


def p_other(p):
    """other : OTHER THAN string
    | empty
    """
    if len(p) > 2:
        p[0] = p[3]


def p_order_by(p):
    """order_by : ORDER BY order
    | empty
    """
    p[0] = []
    if len(p) > 2:
        p[0] = p[3]


def p_limit(p):
    """limit : LIMIT NUMBER
    | empty
    """
    p[0] = []
    if len(p) > 2:
        p[0] = p[2]


def p_order(p):
    """order : function "(" string ")" order_type
    | string order_type
    """
    if len(p) == 6:
        p[0] = {"name": p[3], "type": p[5], "function": p[1]["name"]}
    else:
        p[0] = {"name": p[1], "type": p[2], "function": None}


def p_order_type(p):
    """order_type : ASC
    | DESC
    | empty
    """
    if p[1] == "DESC":
        p[0] = "DESC"
    else:
        p[0] = "ASC"


def p_compare(p):
    """compare : column COMPARISON item
    | column BETWEEN item AND item
    | column IS null
    """
    if len(p) == 4:
        p[0] = {"name": p[1]["name"], "value": p[3], "compare": p[2]}
    if len(p) == 6:
        p[0] = {"name": p[1]["name"], "value": [p[3], p[5]], "compare": p[2]}


def p_empty(p):
    """empty :"""
    pass


def p_conditions(p):
    """conditions : conditions AND conditions
    | conditions OR conditions
    | "(" conditions ")"
    | FINETUNED
    | NOT FINETUNED
    | compare
    """
    if len(p) == 2 and p[1] == "FINETUNED":
        p[0] = ["FINETUNED"]
    elif len(p) == 3 and p[2] == "FINETUNED":
        p[0] = ["NOT FINETUNED"]
    elif len(p) == 2:
        p[0] = [p[1]]
    else:
        if "(" in p:
            p[0] = [p[2]]
        else:
            p[0] = p[1] + [p[2]] + p[3]


def p_columns(p):
    """columns : columns COMMA columns
    | column_as
    | column
    | "*"
    | list_columns
    | "(" columns ")"
    """
    if len(p) > 2:
        if "(" in p:
            p[0] = p[2]
        else:
            p[0] = p[1] + p[3]
    else:
        p[0] = [p[1]]


def p_integers(p):
    """integers : integers COMMA integers
    | NUMBER
    """
    if len(p) > 2:
        p[0] = p[1] + p[3]
    else:
        p[0] = [p[1]]


def p_list_integers(p):
    """list_integers : "[" integers "]" """
    p[0] = p[2]


def p_list_columns(p):
    """list_columns : "[" columns "]" """
    p[0] = {"name": [each["name"] for each in p[2]]}


def p_column_as(p):
    """column_as : column AS item
    | column item
    """
    p[0] = p[1]
    if len(p) > 3:
        p[0]["alias"] = p[3]
    else:
        p[0]["alias"] = p[2]


def p_column(p):
    """column : distinct_item
    | item
    """
    if len(p) > 2:
        p[0] = {"name": {p[1]: p[3]}}
    else:
        p[0] = {"name": p[1]}


"""
Below are train/test related tokens.
"""


def p_trained_on(p):
    """trained_on : TRAINED ON datasets
    | empty
    """
    p[0] = []
    if len(p) > 2:
        p[0] = {"datasets": p[3]}


def p_tested_on(p):
    """tested_on : TESTED ON task_type datasets
    | empty
    """
    p[0] = []
    if len(p) > 3:
        p[0] = {"task_type": p[3], "datasets": p[4]}


def p_task_type(p):
    """task_type : BENCHMARK
    | TASK
    | empty
    """
    p[0] = p[1]


def p_change(p):
    """change :  CHANGE list_integers TO dataset list_integers
    | change COMMA change
    | empty
    """
    if len(p) == 6:
        p[0] = [{"base_indices": p[2], "target": p[4], "change_indices": p[5]}]
    if len(p) == 4:
        p[0] = p[1] + p[3]


def p_datasets(p):
    """datasets : dataset
    | datasets COMMA datasets
    | "[" datasets "]"
    """
    if len(p) > 2:
        if p[1] == "[":
            # for [ datasets dataset, ... ]
            p[0] = p[2]
        else:
            # for dataset, dataset, ...
            p[0] = p[1] + p[3]
    else:
        p[0] = [p[1]]


def p_dataset(p):
    """dataset : string
    | "(" dataset change ")"
    """
    if len(p) == 5:
        p[0] = {"name": p[2]["name"], "changes": p[3]}
    elif len(p) == 2:
        p[0] = {"name": p[1], "changes": []}


def p_classifiers(p):
    """classifiers : "[" classifier "]"
    | classifier
    """
    if len(p) > 3:
        p[0] = p[2]
    else:
        p[0] = p[1]


def p_classifier(p):
    """classifier : classifier COMMA classifier
    | string
    | string "(" classifier_param ")"
    """
    if len(p) > 2:
        if len(p) == 5:
            p[0] = [{"classifier": p[1], "params": p[3]}]
        else:
            p[0] = p[1] + p[3]

    else:
        p[0] = [p[1]]


def p_classifier_param(p):
    """classifier_param : classifier_param COMMA classifier_param
    | item COMPARISON item
    """
    if p[2] == "=":
        p[0] = {p[1]: p[3]}
    elif p[2] == ",":
        p[0] = {**p[1], **p[3]}


def p_classified_by(p):
    """classified_by : CLASSIFIED BY classifiers
    | empty
    """
    p[0] = []
    if len(p) > 2:
        p[0] = p[3]


"""
Below are utilities for the parser
"""


def p_null(p):
    """null : NULL
    | NOT NULL
    """
    if len(p) == 2:
        p[0] = "NULL"
    else:
        p[0] = "NOT NULL"


def p_function(p):
    """function : SUM
    | AVG
    | MIN
    | MAX
    """
    p[0] = {"name": p[1]}


def p_distinct_item(p):
    """distinct_item : DISTINCT item
    | DISTINCT "(" item ")"
    """
    if len(p) > 3:
        p[0] = {p[1]: p[3]}
    else:
        p[0] = {p[1]: p[2]}


def p_item(p):
    """item : string
    | NUMBER
    | "*"
    | string "." item
    """
    if len(p) > 2:
        p[0] = p[1] + "." + p[3]
    else:
        p[0] = p[1]


def p_error(p):
    print("Error occurred at {}".format(p))
    raise GrammarException("Syntax error!")
