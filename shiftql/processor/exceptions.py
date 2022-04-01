class LexerException(Exception):
    pass


class GrammarException(Exception):
    pass


class KeywordNotFoundException(Exception):
    pass


class UnableToCompareException(Exception):
    pass


class TableNotExistsException(Exception):
    pass


class ImproperUsageException(Exception):
    pass


class ServerError(Exception):
    pass
