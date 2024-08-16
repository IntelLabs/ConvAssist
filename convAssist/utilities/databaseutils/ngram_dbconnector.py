# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# from __future__ import absolute_import, unicode_literals

import re
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple
from ConvAssist.utilities.logging import ConvAssistLogger
from ConvAssist.utilities.databaseutils.dbconnector import DatabaseConnector, DatabaseError

re_escape_singlequote = re.compile("'")


class NGramDatabaseConnector(DatabaseConnector):
    """
    Abstract base class for database interactions.
    """
    def __init__(self, dbname: str, cardinality=1, logger=None):
        self.cardinality = cardinality
        self.dbname = dbname
        self.lowercase = False
        self.normalize = False
        self.connection = None
        
        if logger:
            self.log = logger
        else:
            self.log = ConvAssistLogger(name="DatabaseConnector", 
                                        level=ConvAssistLogger.DEBUG)

    @abstractmethod
    def connect(self, **kwargs) -> None:
        """
        Establish a connection to the database.
        """
        pass # pragma: no cover

    @abstractmethod
    def close(self) -> None:
        """
        Close the database connection.
        """
        pass # pragma: no cover

    @abstractmethod
    def execute_query(self, query: str, params: Optional[Tuple[Any, ...]] = None) -> None:
        """
        Execute a query without returning any results.
        """
        pass # pragma: no cover

    @abstractmethod
    def fetch_one(self, query: str, params: Optional[Tuple[Any, ...]] = None) -> Optional[Tuple[Any, ...]]:
        """
        Execute a query and return a single result.
        """
        pass # pragma: no cover

    @abstractmethod
    def fetch_all(self, query: str, params: Optional[Tuple[Any, ...]] = None) -> List[Tuple[Any, ...]]:
        """
        Execute a query and return all results.
        """
        pass # pragma: no cover

    @abstractmethod
    def begin_transaction(self) -> None:
        """
        Begin a new database transaction.
        """
        pass # pragma: no cover

    @abstractmethod
    def commit(self) -> None:
        """
        Commit the current transaction.
        """
        pass # pragma: no cover

    @abstractmethod
    def rollback(self) -> None:
        """
        Roll back the current transaction.
        """
        pass # pragma: no cover
    
    # Implemeneted NGRAM Functionality
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
        query = "CREATE TABLE IF NOT EXISTS _{0}_gram (".format(cardinality)
        unique = ""
        for i in reversed(range(cardinality)):
            if i != 0:
                unique += "word_{0}, ".format(i)
                query += "word_{0} TEXT, ".format(i)
            else:
                unique += "word"
                query += "word TEXT, count INTEGER, UNIQUE({0}) );".format(unique)

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

        query = "DROP TABLE IF EXISTS _{0}_gram;".format(cardinality)
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
                query = "DROP INDEX IF EXISTS idx_{0}_gram_{1};".format(cardinality, i)
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
                query += "word_{0}, ".format(i)
            elif i == 0:
                query += "word"

        if with_counts:
            query += ", count"

        query += " FROM _{0}_gram;".format(self.cardinality)

        result = self.execute_query(query)
        for row in result:
            yield tuple(row)

    def unigram_counts_sum(self):
        query = "SELECT SUM(count) from _1_gram;"
        result = self.execute_query(query)
        if(result==[(None,)]):
            return [(0,)]
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
        query = "SELECT count FROM _{0}_gram".format(len(ngram))
        query += self._build_where_clause(ngram)
        query += ";"
        result = self.execute_query(query)

        return self._extract_first_integer(result)

    def ngram_like_table(self, ngram, limit=-1):
        query = "SELECT {0} FROM _{1}_gram {2} ORDER BY count DESC".format(
            self._build_select_like_clause(len(ngram)),
            len(ngram),
            self._build_where_like_clause(ngram),
        )
        if limit < 0:
            query += ";"
        else:
            query += " LIMIT {0};".format(limit)

        return self.execute_query(query)

    def ngram_like_table_filtered(self, ngram, filter, limit=-1):
        pass

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
        query_check = f"SELECT * from _{len(ngram)}_gram where word = '{ngram[0]}';"
        query_insert = f"INSERT INTO _{len(ngram)}_gram {values});"
        
        # query = "INSERT INTO _{0}_gram {1};".format(
        #     len(ngram), self._build_values_clause(ngram, count)
        # )
        if self.execute_query(query_check) is None:
            try:
                self.execute_query(query_insert)
            except Exception as e:
                print(f"Exception while processing this sql query: {query_insert}")
                raise e
        else:
            print(f"Word '{ngram[0]}' already exists in the database.")
            pass

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
        query = "UPDATE _{0}_gram SET count = {1}".format(len(ngram), count)
        query += self._build_where_clause(ngram)
        query += ";"
        self.execute_query(query)

    def remove_ngram(self, ngram):
        """
        Removes a given ngram from the databae. The ngram has to be in the
        database, otherwise this method will stop with an error.

        Parameters
        ----------
        ngram : iterable of str
            A list, set or tuple of strings.

        """
        query = "DELETE FROM _{0}_gram".format(len(ngram))
        query += self._build_where_clause(ngram)
        query += ";"
        self.execute_query(query)

    def _build_values_clause(self, ngram, count):
        ngram_escaped = []
        for n in ngram:
            ngram_escaped.append(re_escape_singlequote.sub("''", n))

        values_clause = "VALUES('"
        values_clause += "', '".join(ngram_escaped)
        values_clause += "', {0})".format(count)
        return values_clause

    def _build_where_clause(self, ngram):
        where_clause = " WHERE"
        for i in range(len(ngram)):
            n = re_escape_singlequote.sub("''", ngram[i])
            if i < (len(ngram) - 1):
                where_clause += " word_{0} = '{1}' AND".format(len(ngram) - i - 1, n)
            else:
                where_clause += " word = '{0}'".format(n)
        return where_clause

    def _build_select_like_clause(self, cardinality):
        result = ""
        for i in reversed(range(cardinality)):
            if i != 0:
                result += "word_{0}, ".format(i)
            else:
                result += "word, count"
        return result

    def _build_where_like_clause(self, ngram):
        re_escape_singlequote = re.compile(r"(')")

        where_clause = " WHERE"
        escaped_ngram = [re_escape_singlequote.sub("''", item) for item in ngram]

        for i, item in enumerate(escaped_ngram):
            if i < (len(escaped_ngram) - 1):
                where_clause += " word_{0} = '{1}' AND".format(
                    len(escaped_ngram) - i - 1, item
                )
            else:
                where_clause += " word LIKE '{0}%'".format(item)
        return where_clause

    def _extract_first_integer(self, table):
        count = 0
        if len(table) > 0:
            if len(table[0]) > 0:
                count = int(table[0][0])

        if not count > 0:
            count = 0
        return count

