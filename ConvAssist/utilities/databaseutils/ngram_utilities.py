# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: GPL-3.0-or-later

import re
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple

from ConvAssist.utilities.databaseutils.dbconnector import (
    DatabaseConnector,
    DatabaseError,
)

re_escape_singlequote = re.compile("'")


class NGramUtilities(DatabaseConnector):
    """
    Abstract base class for database interactions.
    """

    def __init__(self, cardinality=1, logger=None):
        super().__init__(logger)
        self.cardinality = cardinality
        self.lowercase = False
        self.normalize = False
        self.connection = None

    # Implemented NGRAM Functionality
    def create_ngram_table(self, cardinality):
        """
        Creates a table for n-gram of a given cardinality. The table name is
        constructed from this parameter, for example for cardinality `2` there
        will be a table `_2_gram` created.

        Parameters
        ----------
        cardinality : int
            The cardinality to create a table for.

        """
        # TODO Refactor to use dbconnector.create_table
        query = f"CREATE TABLE IF NOT EXISTS _{cardinality}_gram ("
        unique = ""
        for i in reversed(range(cardinality)):
            if i != 0:
                unique += f"word_{i}, "
                query += f"word_{i} TEXT, "
            else:
                unique += "word"
                query += f"word TEXT, count INTEGER, UNIQUE({unique}) );"

        self.execute_query(query)

    def delete_ngram_table(self, cardinality):
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
        self.execute_query(query)

    def create_index(self, cardinality):
        """
        Create an index for the table with the given cardinality.

        Parameters
        ----------
        cardinality : int
            The cardinality to create a index for.

        """
        for i in reversed(range(cardinality)):
            if i != 0:
                query = "CREATE INDEX idx_{0}_gram_{1} ON _{0}_gram(word_{1});".format(
                    cardinality, i
                )
                self.execute_query(query)

    def delete_index(self, cardinality):
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
                self.execute_query(query)

    def create_unigram_table(self):
        """
        Creates a table for n-grams of cardinality 1.

        """
        self.create_ngram_table(1)

    def create_bigram_table(self):
        """
        Creates a table for n-grams of cardinality 2.

        """
        self.create_ngram_table(2)

    def create_trigram_table(self):
        """
        Creates a table for n-grams of cardinality 3.

        """
        self.create_ngram_table(3)

    def ngrams(self, with_counts=False):
        """
        Returns all ngrams that are in the table.

        Parameters
        ----------
        None

        Returns
        -------
        ngrams : generator
            A generator for ngram tuples.

        """
        query = "SELECT "
        for i in reversed(range(self.cardinality)):
            if i != 0:
                query += f"word_{i}, "
            elif i == 0:
                query += "word"

        if with_counts:
            query += ", count"

        query += f" FROM _{self.cardinality}_gram;"

        result = self.fetch_all(query)
        if result:
            for row in result:
                yield tuple(row)

    def unigram_counts_sum(self) -> int:
        query = "SELECT SUM(count) from _1_gram;"
        result = self.fetch_all(query)
        if result == [(None,)]:
            return 0
        return self._extract_first_integer(result)

    def ngram_count(self, ngram):
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
        result = self.fetch_all(query, tuple(ngram))

        return self._extract_first_integer(result)

    def ngram_fetch_like(self, ngram, limit=-1):
        try:
            query = "SELECT {} FROM _{}_gram {} ORDER BY count DESC".format(  # nosec
                self._build_select_like_clause(len(ngram)),
                len(ngram),
                self._build_where_like_clause(ngram),
            )
            if limit < 0:
                query += ";"
            else:
                query += f" LIMIT {limit};"

            result = self.fetch_all(query)
        except DatabaseError as e:
            self.logger.critical(f"Error while ngram_fetch_like this query: {query}")
            raise e
        return result

    def increment_ngram_count(self, ngram):
        pass

    def insert_ngram(self, ngram, count):
        """
        Inserts a given n-gram with count into the database.

        Parameters
        ----------
        ngram : iterable of str
            A list, set or tuple of strings.
        count : int
            The count for the given n-gram.

        """
        values = self._build_values_clause(ngram, count)
        escaped_ngram = re_escape_singlequote.sub("''", ngram[0])
        query_check = f"SELECT * from _{len(ngram)}_gram where word = '{escaped_ngram}';"  # nosec
        query_insert = f"INSERT INTO _{len(ngram)}_gram {values};"

        try:
            if self.fetch_all(query_check) is None:
                self.execute_query(query_insert)
            else:
                self.logger.info(f"Word '{ngram[0]}' already exists in the database.")
                pass
        except Exception as e:
            self.logger.critical(f"Exception while processing this sql query: {query_insert}")
            raise e

    def update_ngram(self, ngram, count):
        """
        Updates a given ngram in the database. The ngram has to be in the
        database, otherwise this method will stop with an error.

        Parameters
        ----------
        ngram : iterable of str
            A list, set or tuple of strings.
        count : int
            The count for the given n-gram.

        """
        query = f"UPDATE _{len(ngram)}_gram SET count = {count}"
        query += self._build_where_clause(ngram)
        query += ";"
        self.execute_query(query)

    def remove_ngram(self, ngram):
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
        self.execute_query(query)

    def _build_values_clause(self, ngram, count):
        ngram_escaped = []
        for n in ngram:
            ngram_escaped.append(re_escape_singlequote.sub("''", n))

        values_clause = "VALUES('{}', {})".format("', '".join(ngram_escaped), count)
        return values_clause

    def _build_where_clause(self, ngram):
        where_clause = " WHERE"
        for i in range(len(ngram)):
            if i < (len(ngram) - 1):
                where_clause += f" word_{len(ngram) - i - 1} = ? AND"
            else:
                where_clause += " word = ?"
        return where_clause

    def _build_select_like_clause(self, cardinality):
        result = ""
        for i in reversed(range(cardinality)):
            if i != 0:
                result += f"word_{i}, "
            else:
                result += "word, count"
        return result

    def _build_where_like_clause(self, ngram):
        re_escape_singlequote = re.compile(r"(')")

        where_clause = " WHERE"
        escaped_ngram = [re_escape_singlequote.sub("''", item) for item in ngram]

        for i, item in enumerate(escaped_ngram):
            if i < (len(escaped_ngram) - 1):
                where_clause += f" word_{len(escaped_ngram) - i - 1} = '{item}' AND"
            else:
                where_clause += f" word LIKE '{item}%' ESCAPE '\\'"
        return where_clause

    def _extract_first_integer(self, table):
        count = 0
        if table and len(table) > 0:
            if len(table[0]) > 0:
                count = int(table[0][0])

        if not count > 0:
            count = 0
        return count
