# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import sqlite3
from typing import Any, Optional, Tuple, List
from ConvAssist.utilities.databaseutils.dbconnector import DatabaseError, DatabaseConnector

class SQLiteDatabaseConnector(DatabaseConnector):
    def __init__(self, dbname: str, logger=None):
        super().__init__(logger=logger)
        self.dbname = dbname
        self.conn:sqlite3.Connection | None = None

    def connect(self, **kwargs) -> sqlite3.Connection:
        self.logger.debug(f"Connecting to SQLite database {self.dbname}")
        self.conn = sqlite3.connect(self.dbname)
        return self.conn

    def close(self) -> None:
        if self.conn:
            self.conn.close()
            self.conn = None

    def execute_query(self, query: str, params: Optional[Tuple[Any, ...]] = None) -> None:
        if not self.conn:
            raise DatabaseError("Database connection is not established.")
        
        cursor = self.conn.cursor()

        try:
            cursor.execute(query, params or ())
            self.conn.commit()

        finally:
            cursor.close()  
  
    def fetch_one(self, query: str, params: Optional[Tuple[Any, ...]] = None) -> Optional[Tuple[Any, ...]] | None:
        if not self.conn:
            raise DatabaseError("Database connection is not established.")

        cursor = self.conn.cursor()
        try:
            cursor.execute(query, params or ())
            return cursor.fetchone()
        
        finally:
            cursor.close()

    def fetch_all(self, query: str, params: Optional[Tuple[Any, ...]] = None) -> List[Tuple[Any, ...]] | None:
        if not self.conn:
            raise DatabaseError("Database connection is not established.")
        
        cursor = self.conn.cursor()
        try:
            cursor.execute(query, params or ())
            return cursor.fetchall()
        
        finally:
            cursor.close()

    def begin_transaction(self) -> None:
        if not self.conn:
            raise DatabaseError("Database connection is not established.")
        self.conn.execute('BEGIN')

    def commit(self) -> None:
        if not self.conn:
            raise DatabaseError("Database connection is not established.")
        self.conn.commit()

    def rollback(self) -> None:
        if not self.conn:
            raise DatabaseError("Database connection is not established.")
        self.conn.rollback()

