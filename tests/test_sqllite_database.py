import sqlite3
from pathlib import Path
import unittest
# from unittest.mock import MagicMock
from src.utilities.databaseutils.sqllite_dbconnector import SQLiteDatabaseConnector
from tests.utils import safe_delete_file, safe_check_folder

class TestSQLiteDatabase(unittest.TestCase):
    def setUp(self):
        self.db_path = "tests/test_data/dbs/"
        self.db_file = "test_sqllite.db"
        safe_check_folder(self.db_path)
        self.db = SQLiteDatabaseConnector(str(Path(self.db_path) / (self.db_file)))

    def tearDown(self):
        self.db.close()
        safe_delete_file(str(Path(self.db_path) / (self.db_file)))

    def test_connect(self):
        self.db.connect()
        self.assertIsNotNone(self.db.conn)

    def test_close(self):
        self.db.connect()
        self.db.close()
        self.assertIsNone(self.db.conn)   

    def test_execute_query(self):
        self.db.connect()
        query = "CREATE TABLE IF NOT EXISTS test_table (id INTEGER PRIMARY KEY, name TEXT)"
        self.db.execute_query(query)
        result = self.db.fetch_all("SELECT name FROM sqlite_master WHERE type='table' AND name='test_table'")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0], "test_table")

    def test_begin_transaction(self):
        self.db.connect()
        self.db.begin_transaction()
        self.assertIsNotNone(self.db.conn)

    def test_commit(self):
        self.db.connect()
        self.db.begin_transaction()
        self.db.commit()
        self.assertIsNotNone(self.db.conn)

    def test_rollback(self):
        self.db.connect()
        self.db.begin_transaction()
        self.db.rollback()
        self.assertIsNotNone(self.db.conn)

class TestSQLiteFetchCommands(unittest.TestCase):
    def setUp(self):
        self.db_path = "tests/test_data/dbs/"
        self.db_file = "test_sqllite_fetch.db"
        safe_check_folder(self.db_path)
        self.db = SQLiteDatabaseConnector(str(Path(self.db_path) / (self.db_file)))
        self.db.connect()
        query = "CREATE TABLE IF NOT EXISTS test_table (id INTEGER PRIMARY KEY, name TEXT)"
        self.db.execute_query(query)

    def tearDown(self):
        self.db.close()
        safe_delete_file(str(Path(self.db_path) / (self.db_file)))

    def test_fetch_one(self):
        self.db.connect()
        query = "SELECT COUNT(*) FROM test_table"
        result = self.db.fetch_one(query)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], int)

    def test_fetch_all(self):
        self.db.connect()
        query = "SELECT * FROM test_table"
        result = self.db.fetch_all(query)
        self.assertIsInstance(result, list)

if __name__ == "__main__":
    unittest.main()