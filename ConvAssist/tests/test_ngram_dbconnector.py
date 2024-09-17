# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: GPL-3.0-or-later

import unittest
from unittest.mock import MagicMock
from ConvAssist.utilities.databaseutils.sqllite_ngram_dbconnector import SQLiteNgramDatabaseConnector

class TestNGramDatabaseConnector(unittest.TestCase):
    def setUp(self):
        # Create a mock instance of NGramDatabaseConnector
        self.connector = SQLiteNgramDatabaseConnector(dbname="test_db")
        self.connector.execute_query = MagicMock()

    def test_create_ngram_table(self):
        # Test for cardinality 1
        self.connector.create_ngram_table(1)
        expected_query_1 = "CREATE TABLE IF NOT EXISTS _1_gram (word TEXT, count INTEGER, UNIQUE(word) );"
        self.connector.execute_query.assert_called_with(expected_query_1)

        # Test for cardinality 2
        self.connector.create_ngram_table(2)
        expected_query_2 = "CREATE TABLE IF NOT EXISTS _2_gram (word_1 TEXT, word TEXT, count INTEGER, UNIQUE(word_1, word) );"
        self.connector.execute_query.assert_called_with(expected_query_2)

        # Test for cardinality 3
        self.connector.create_ngram_table(3)
        expected_query_3 = "CREATE TABLE IF NOT EXISTS _3_gram (word_2 TEXT, word_1 TEXT, word TEXT, count INTEGER, UNIQUE(word_2, word_1, word) );"
        self.connector.execute_query.assert_called_with(expected_query_3)

if __name__ == "__main__":
    unittest.main()
