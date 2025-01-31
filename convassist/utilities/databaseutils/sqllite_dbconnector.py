# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: GPL-3.0-or-later

import sqlite3

# import multiprocessing
import threading
from typing import Any, List, Optional, Tuple

from convassist.utilities.databaseutils.dbconnector import DatabaseConnector, DatabaseError


class SQLiteDatabaseConnector(DatabaseConnector):
    def __init__(self, dbname: str):
        super().__init__()
        self.dbname = dbname
        self.conn: Optional[sqlite3.Connection] = None
        self.lock = threading.Lock()
        self.connect()

    def connect(self) -> None:
        with self.lock:
            if self.conn:
                # Connection already established
                return
            else:
                self.conn = sqlite3.connect(self.dbname, check_same_thread=False)
                self.conn.execute("PRAGMA busy_timeout = 5000")  # 5 seconds

    def execute_query(self, query: str, params: Optional[Tuple[Any, ...]] = None) -> None:
        with self.lock:
            if not self.conn:
                raise DatabaseError("Database connection is not established.")
            cursor = self.conn.cursor()
            try:
                cursor.execute(query, params or ())
                self.conn.commit()
            except Exception as e:
                raise DatabaseError(f"Error executing query: {e}")
            finally:
                cursor.close()

    def execute_many(self, query: str, params: List[Tuple[Any, ...]]) -> None:
        with self.lock:
            if not self.conn:
                raise DatabaseError("Database connection is not established.")
            cursor = self.conn.cursor()
            try:
                cursor.executemany(query, params)
                self.conn.commit()
            except Exception as e:
                self.conn.rollback()
                raise DatabaseError(f"Error executing query: {e}")
            finally:
                cursor.close()

    def fetch_one(
        self, query: str, params: Optional[Tuple[Any, ...]] = None
    ) -> Optional[Tuple[Any, ...]]:
        with self.lock:
            if not self.conn:
                raise DatabaseError("Database connection is not established.")
            cursor = self.conn.cursor()
            try:
                cursor.execute(query, params or ())
                result = cursor.fetchone()
            except Exception as e:
                raise DatabaseError(f"Error fetching data from database: {e}")
            finally:
                cursor.close()
            return result

    def fetch_all(
        self, query: str, params: Optional[Tuple[Any, ...]] = None
    ) -> List[Tuple[Any, ...]]:
        with self.lock:
            if not self.conn:
                raise DatabaseError("Database connection is not established.")
            cursor = self.conn.cursor()
            try:
                cursor.execute(query, params or ())
                result = cursor.fetchall()
            except Exception as e:
                raise DatabaseError(f"Error fetching data from database: {e}")
            finally:
                cursor.close()
            return result

    def close(self) -> None:
        with self.lock:
            if self.conn:
                self.conn.close()
                self.conn = None

    def create_table(self, tablename: str, columns: List[str]) -> None:
        query = f"CREATE TABLE IF NOT EXISTS {tablename} ({', '.join(columns)})"
        self.execute_query(query)

    def check_write_access(self) -> bool:
        try:
            self.execute_query("CREATE TABLE temp_table (id INTEGER)")
            self.execute_query("DROP TABLE temp_table")
            return True

        except DatabaseError:
            return False
