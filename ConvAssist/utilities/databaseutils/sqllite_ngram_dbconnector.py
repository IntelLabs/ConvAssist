# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: GPL-3.0-or-later

from .ngram_utilities import NGramUtilities
from .sqllite_dbconnector import SQLiteDatabaseConnector


class SQLiteNgramDatabaseConnector(SQLiteDatabaseConnector, NGramUtilities):
    """
    SQLiteNgramDatabaseConnector is a class that connects to a SQLite database and
    provides methods to create and query n-gram tables.
    """

    def __init__(self, dbname: str, cardinality=1, logger=None):
        SQLiteDatabaseConnector.__init__(self, dbname=dbname, logger=logger)
        NGramUtilities.__init__(self, cardinality=cardinality, logger=logger)

    """
    All the methods from NGramDatabaseConnector and SQLiteDatabaseConnector are inherited
    """
