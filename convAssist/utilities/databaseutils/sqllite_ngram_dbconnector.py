# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import sqlite3
from typing import Any, Optional, Tuple, List
from ConvAssist.utilities.databaseutils.ngram_dbconnector import NGramDatabaseConnector, DatabaseError

class SQLiteNgramDatabaseConnector(NGramDatabaseConnector):
    def __init__(self, dbname: str, cardinality=1, logger=None):
        super().__init__(dbname=dbname, cardinality=cardinality, logger=logger)

    def connect(self) -> None:
        self.log.debug(f"Connecting to SQLite database {self.dbname}")
        self.connection = sqlite3.connect(self.dbname)

    def close(self) -> None:
        if self.connection:
            self.connection.close()
            self.connection = None

    def execute_query(self, query: str, params: Optional[Tuple[Any, ...]] = None) -> None:
        if not self.connection:
            raise DatabaseError("Database connection is not established.")
        cursor = self.connection.cursor()
        cursor.execute(query, params or ())
        self.connection.commit()
        cursor.close()

    def fetch_one(self, query: str, params: Optional[Tuple[Any, ...]] = None) -> Optional[Tuple[Any, ...]]:
        if not self.connection:
            raise DatabaseError("Database connection is not established.")
        cursor = self.connection.cursor()
        cursor.execute(query, params or ())
        result = cursor.fetchone()
        cursor.close()
        return result

    def fetch_all(self, query: str, params: Optional[Tuple[Any, ...]] = None) -> List[Tuple[Any, ...]]:
        if not self.connection:
            raise DatabaseError("Database connection is not established.")
        cursor = self.connection.cursor()
        cursor.execute(query, params or ())
        results = cursor.fetchall()
        cursor.close()
        return results

    def begin_transaction(self) -> None:
        if not self.connection:
            raise DatabaseError("Database connection is not established.")
        self.connection.execute('BEGIN')

    def commit(self) -> None:
        if not self.connection:
            raise DatabaseError("Database connection is not established.")
        self.connection.commit()

    def rollback(self) -> None:
        if not self.connection:
            raise DatabaseError("Database connection is not established.")
        self.connection.rollback()

