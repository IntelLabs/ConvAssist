# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: GPL-3.0-or-later

import re
from typing import List, Optional

from convassist.utilities.databaseutils.sqllite_dbconnector import (
    SQLiteDatabaseConnector,
)
from convassist.utilities.ngram.ngram_map import NgramMap

re_escape_singlequote = re.compile("'")


class NGramUtil:
    def __init__(self, database, cardinality=1, lowercase=False, normalize=False):
        self._database = database
        self._cardinality = cardinality
        self._lowercase = lowercase
        self._normalize = normalize
        self._connection = SQLiteDatabaseConnector(database)

    def __enter__(self):
        try:
            self._connection.connect()
            self._create_card_tables()
        except Exception as e:
            raise Exception(f"{__class__}{__name__} failed to connect to db: {e}")

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._connection:
            self._connection.close()

    # Implemented NGRAM Functionality
    def _create_ngram_table(self, cardinality):
        """
        Creates a table for n-gram of a given cardinality. The table name is
        constructed from this parameter, for example for cardinality `2` there
        will be a table `_2_gram` created.

        Parameters
        ----------
        cardinality : int
            The cardinality to create a table for.

        """
        columns = []
        unique = ""
        for i in reversed(range(cardinality)):
            if i != 0:
                columns.append(f"word_{i} TEXT")
                unique = ", ".join([f"word_{i}", unique])
            else:
                columns.append("word TEXT")
                unique = "".join([unique, "word"])
        columns.append("count INTEGER")

        # unique = ", ".join([f"word_{i}" for i in reversed(range(cardinality - 1))]) if cardinality > 1 else ""
        # unique = ", ".join([unique, "word"]) if cardinality > 1 else "word"
        columns.append(f"UNIQUE({unique})")

        self._connection.create_table(f"_{cardinality}_gram", columns)

    def _delete_ngram_table(self, cardinality):
        """
        Deletes the table for n-gram of a give cardinality. The table name is
        constructed from this parameter, for example for cardinality `2` there
        will be a table `_2_gram` deleted.

        Parameters
        ----------
        cardinality : int
            The cardinality of the table to delete.

        """

        query = f"DROP TABLE IF EXISTS _{cardinality}_gram;"
        self._connection.execute_query(query)

    def _create_index(self, cardinality):
        """
        Create an index for the table with the given cardinality.

        Parameters
        ----------
        cardinality : int
            The cardinality to create a index for.

        """
        for i in reversed(range(cardinality)):
            if i != 0:
                query = (
                    "CREATE INDEX IF NOT EXISTS idx_{0}_gram_{1} ON _{0}_gram(word_{1});".format(
                        cardinality, i
                    )
                )
                self._connection.execute_query(query)

    def _delete_index(self, cardinality):
        """
        Delete index for the table with the given cardinality.

        Parameters
        ----------
        cardinality : int
            The cardinality of the index to delete.

        """
        for i in reversed(range(cardinality)):
            if i != 0:
                query = f"DROP INDEX IF EXISTS idx_{cardinality}_gram_{i};"
                self._connection.execute_query(query)

    def _ngram_count(self, ngram):
        """
        Gets the count for a given ngram from the database.

        Parameters
        ----------
        ngram : iterable of str
            A list, set or tuple of strings.

        Returns
        -------
        count : int
            The count of the ngram.

        """
        query = f"SELECT count FROM _{len(ngram)}_gram"  # nosec
        query += self._build_where_clause(ngram)
        query += ";"
        result = self._connection.fetch_all(query, tuple(ngram))

        return self._extract_first_integer(result)

    def _insert_ngram(self, cardinality, ngram, count, update_on_conflict=True):
        """
        Inserts a given n-gram with count into the database.

        Parameters
        ----------
        ngram : iterable of str
            A list, set or tuple of strings.
        count : int
            The count for the given n-gram.

        INSERT INTO _{N}_gram (word_{N}, word_{N-1}, word)
        VALUES ('V1', 'V2', 'V3', count)
        ON CONFLICT (word_{N}, word_{N-1}, word)
        DO UPDATE SET count

        """
        table_name = f"_{cardinality}_gram"
        columns = ", ".join([f"word_{i + 1}" for i in reversed(range(cardinality - 1))])
        if columns:
            columns = ", ".join([columns, "word"])
        else:
            columns = "word"
        placeholders = ", ".join(["?" for _ in range(cardinality)])
        unique_columns = columns

        query = f"INSERT INTO {table_name} ({columns}, count) VALUES ({placeholders}, ?)"
        if update_on_conflict:
            query += f"ON CONFLICT ({unique_columns}) DO UPDATE SET count = count + 1;"
        else:
            query += f"ON CONFLICT ({unique_columns}) DO NOTHING;"

        try:
            self._connection.execute_query(query, (*ngram, count))
            self._connection.commit()
        except Exception as e:
            raise Exception(f"{__class__}{__name__} failed to create ngram table: {e}")

    def _remove_ngram(self, ngram):
        """
        Removes a given ngram from the database. The ngram has to be in the
        database, otherwise this method will stop with an error.

        Parameters
        ----------
        ngram : iterable of str
            A list, set or tuple of strings.

        """
        query = f"DELETE FROM _{len(ngram)}_gram"  # nosec
        query += self._build_where_clause(ngram)
        query += ";"
        self._connection.execute_query(query, (ngram))

    def _build_values_clause(self, ngram, count):
        ngram_escaped = []
        for n in ngram:
            ngram_escaped.append(re_escape_singlequote.sub("''", n))

        values_clause = "VALUES('{}', {})".format("', '".join(ngram_escaped), count)

        return values_clause

    def _build_update_where_clause(self, ngram):
        where_clause = " WHERE"
        for i in range(len(ngram)):
            if i < (len(ngram) - 1):
                where_clause += f" word_{len(ngram) - i - 1} = {ngram[i]} AND"
            else:
                where_clause += f" word = {ngram[i]}"
        return where_clause

    def _build_where_clause(self, ngram):
        where_clause = " WHERE"
        for i in range(len(ngram)):
            if i < (len(ngram) - 1):
                where_clause += f" word_{len(ngram) - i - 1} = ? AND"
            else:
                where_clause += " word LIKE ?"
        return where_clause

    def _extract_first_integer(self, table):
        count = 0
        if table and len(table) > 0:
            if len(table[0]) > 0:
                count = int(table[0][0])

        if not count > 0:
            count = 0
        return count

    def _create_card_tables(self):
        # Check if we have write access.  Fail silently if we don't.
        if not self._connection.check_write_access():
            return

        try:
            for i in range(self._cardinality):
                self._create_ngram_table(cardinality=i + 1)
                self._create_index(cardinality=i + 1)

        except Exception as e:
            raise Exception(f"{__class__}{__name__} failed to create ngram table: {e}")

    def count(self, tokens, offset, ngram_size):
        result = 0
        if ngram_size > 0:
            ngram = tokens[len(tokens) - ngram_size + offset : len(tokens) + offset]
            result = self._ngram_count(ngram)
        else:
            result = self.unigram_counts_sum()
        return result

    def _table_exists(self, table_name):
        query = f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}';"
        result = self._connection.fetch_all(query)
        return len(result) > 0

    def unigram_counts_sum(self) -> int:
        return self.counts_sum(1)

    def counts_sum(self, ngram_size):
        query = f"SELECT SUM(count) from _{ngram_size}_gram;"
        result = self._connection.fetch_all(query)
        if result == [(None,)]:
            return 0
        return self._extract_first_integer(result)

    def update(
        self,
        phrases_toAdd: Optional[List[str]] = None,
        phrases_toRemove: Optional[List[str]] = None,
        update_on_conflict: bool = False,
    ):
        assert self._connection is not None

        # Add phrases_toAdd to the ngram database
        if phrases_toAdd:
            for phrase in phrases_toAdd:
                for curr_card in range(1, self._cardinality + 1):
                    ngram_map = NgramMap(curr_card, phrase)

                    # for every ngram, get db count, update or insert
                    for ngram, count in ngram_map.items():
                        self._insert_ngram(curr_card, ngram, count, update_on_conflict)

        if phrases_toRemove:
            for phrase in phrases_toRemove:
                for curr_card in range(self._cardinality):
                    # imp_words = self.extract_svo(phrase)
                    ngram_map = NgramMap(curr_card, phrase)

                    # for every ngram, get db count, update or insert
                    for ngram, count in ngram_map.items():
                        self._remove_ngram(ngram)

    def learn(self, phrase: str):
        assert self._connection is not None

        for card in range(1, self._cardinality + 1):
            ngram_map = NgramMap(card, phrase)

            # write this ngram_map to LM ...
            # for every ngram, get db count, update or insert
            for ngram, count in ngram_map.items():
                self._insert_ngram(card, ngram, count, True)

    def fetch_like(self, ngram: list, limit=-1):
        assert self._connection is not None

        try:
            query: str = ""
            table_name = f"_{self._cardinality}_gram"
            conditions = []
            params = []

            for index, _ in enumerate(ngram):
                inverse_index = len(ngram) - 1 - index

                if index == len(ngram) - 1:
                    conditions.append("word LIKE ?")
                    params.append(f"{ngram[index]}%") if ngram[index] else params.append("%")
                else:
                    conditions.append(f"word_{inverse_index} = ?")
                    params.append(ngram[index])

            where_clause = " AND ".join(conditions)
            query = (
                f"SELECT word, count from {table_name} WHERE {where_clause} ORDER BY count DESC"
            )

            if limit < 0:
                query += ";"
            else:
                query += f" LIMIT {limit};"

            result = self._connection.fetch_all(query, params)
        except Exception as e:
            raise Exception(f"{__class__}{__name__} failed to fetch ngram: {e}")

        return result
