from ply import lex, yacc
from processor.syntax.dml import *

DEBUG = True

import processor.lexer as lexer

tokens = lexer.tokens


class Parser(object):
    def __init__(self, debug=False) -> None:
        super().__init__()
        self.debug = debug
        self.L = lex.lex(module=lexer, optimize=False, debug=debug)
        self.P = yacc.yacc(debug=debug)

    def parse(self, stmt):
        return self.P.parse(input=stmt, lexer=self.L, debug=self.debug)
