# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: GPL-3.0-or-later

import sqlite3
from typing import Any, List, Optional, Tuple

from .dbconnector import DatabaseConnector, DatabaseError


class SQLiteDatabaseConnector(DatabaseConnector):
    def __init__(self, dbname: str):
        super().__init__()
        self.dbname = dbname
        self.conn: sqlite3.Connection | None = None

    def connect(self, **kwargs) -> sqlite3.Connection:
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

        except Exception as e:
            raise DatabaseError(f"Error executing query:{e}")
        finally:
            cursor.close()

    def fetch_one(self, query: str, params: Optional[Tuple[Any, ...]] = None):
        if not self.conn:
            raise DatabaseError("Database connection is not established.")

        result = []
        cursor = self.conn.cursor()
        try:
            cursor.execute(query, params or ())
            result = cursor.fetchone()

        finally:
            cursor.close()
            return result

    def fetch_all(self, query: str, params: Optional[Tuple[Any, ...]] = None):
        if not self.conn:
            raise DatabaseError("Database connection is not established.")

        result = []
        cursor = self.conn.cursor()
        try:
            cursor.execute(query, params or ())
            result = cursor.fetchall()

        except Exception as e:
            raise DatabaseError(f"Error fetching data from database:{e}")

        finally:
            cursor.close()

        return result

    def begin_transaction(self) -> None:
        if not self.conn:
            raise DatabaseError("Database connection is not established.")
        self.conn.execute("BEGIN")

    def commit(self) -> None:
        if not self.conn:
            raise DatabaseError("Database connection is not established.")
        self.conn.commit()

    def rollback(self) -> None:
        if not self.conn:
            raise DatabaseError("Database connection is not established.")
        self.conn.rollback()

    def create_table(self, tablename: str, columns: List[str]) -> None:
        if not self.conn:
            raise DatabaseError("Database connection is not established.")

        try:
            self.execute_query(
                f"""
                CREATE TABLE IF NOT EXISTS {tablename}
                ({', '.join(columns)})
            """
            )
        except Exception as e:
            raise Exception(f"Unable to create table {tablename} in {self.dbname}.", e)

    def check_write_access(self) -> bool:
        try:
            self.execute_query("CREATE TABLE temp_table (id INTEGER)")
            self.execute_query("DROP TABLE temp_table")
            return True

        except DatabaseError:
            return False
