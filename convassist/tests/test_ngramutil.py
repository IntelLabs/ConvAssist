# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: GPL-3.0-or-later

import unittest

from parameterized import parameterized

from convassist.context_tracker import ContextTracker
from convassist.utilities.ngram.ngramutil import NGramUtil


class TestNGramUtil(unittest.TestCase):
    def setUp(self):
        # Create a mock instance of NGramUtil
        self.database = "test_db.db"
        self.cardinality = 1

    def test_init(self):
        # Test the __init__ method of NGramUtil
        with NGramUtil(self.database, self.cardinality) as ngramutil:
            assert ngramutil._database == self.database
            assert ngramutil._cardinality == self.cardinality

            ngramutil.create_update_ngram_tables()
            assert ngramutil._table_exists(f"_{self.cardinality}_gram")

    def test_learn(self):
        # Test the learn method of NGramUtil
        with NGramUtil(":memory:", self.cardinality) as ngramutil:
            ngramutil.create_update_ngram_tables()
            assert ngramutil._table_exists(f"_{self.cardinality}_gram")

            ngramutil.learn("test")
            assert ngramutil.unigram_counts_sum() == 1

    def test_learn_updates(self):
        # Test the learn method of NGramUtil
        with NGramUtil(":memory:", self.cardinality) as ngramutil:

            ngramutil.create_update_ngram_tables()
            assert ngramutil._table_exists(f"_{self.cardinality}_gram")

            ngramutil.learn("test")
            assert ngramutil.unigram_counts_sum() == 1

            ngramutil.learn("test")
            assert ngramutil.unigram_counts_sum() == 2

    def test_fetch_like(self):
        # Test the fetch_like method of NGramUtil
        self.cardinality = 1
        with NGramUtil(":memory:", self.cardinality) as ngramutil:
            ngramutil.create_update_ngram_tables()
            assert ngramutil._table_exists(f"_{self.cardinality}_gram")

            ngramutil.learn("test")
            assert ngramutil.unigram_counts_sum() == 1

            ngramutil.learn("test")
            assert ngramutil.unigram_counts_sum() == 2

            assert ngramutil.fetch_like(["test"], 1) == [("test", 2)]

    def test_fetch_like_multiple(self):
        # Test the fetch_like method of NGramUtil
        self.cardinality = 1
        database = ":memory:"
        # database = "test.db"
        with NGramUtil(database, self.cardinality) as ngramutil:
            ngramutil.create_update_ngram_tables()
            assert ngramutil._table_exists(f"_{self.cardinality}_gram")

            ngramutil.learn("test")
            assert ngramutil.unigram_counts_sum() == 1

            ngramutil.learn("test")
            assert ngramutil.unigram_counts_sum() == 2

            ngramutil.learn("test")
            assert ngramutil.unigram_counts_sum() == 3

            assert ngramutil.fetch_like(["test"], 2) == [("test", 3)]

    def test_fetch_like_explicit_cardinality_preserves_default_and_like_semantics(self):
        with NGramUtil(":memory:", 3) as ngramutil:
            ngramutil.create_update_ngram_tables()
            ngramutil.learn("i like tea")
            ngramutil.learn("you like toast")
            connection = ngramutil.connection.conn

            assert ngramutil.fetch_like(["i", "like", "T"], cardinality=3) == [("tea", 1)]
            assert ngramutil.fetch_like(["like", "t_a"], cardinality=2) == [("tea", 1), ("toast", 1)]
            assert ngramutil.fetch_like(["t%"], 1, cardinality=1) == [("tea", 1)]
            # A shorter prefix still queries the bound table by default, not len(prefix).
            assert ngramutil.fetch_like(["tea"]) == [("tea", 1)]
            assert ngramutil.fetch_like(["i", "like", "tea"], 1) == [("tea", 1)]
            assert ngramutil._cardinality == 3
            assert ngramutil.connection.conn is connection

    @parameterized.expand(
        [
            ("all your bases are mine", "bases are ", "mine", 3),
            ("all your bases are mine", "are ", "mine", 2),
            ("all your bases are mine", "mine", "mine", 1),
            ("it's a beautiful day in the neighborhood", "beautiful day ", "in", 3),
            ("it's beautiful", "it's ", "beautiful", 2),
            ("it's", "it's", "it's", 1),
        ]
    )
    def test_fetch_list_multi_ngram(self, phrase, fetch, expected, card):
        database = ":memory:"
        # database = "test.db"
        with NGramUtil(database, card) as ngramutil:

            for i in range(1, card + 1):
                ngramutil.create_update_ngram_tables()
                assert ngramutil._table_exists(f"_{self.cardinality}_gram")

            ngramutil.learn(phrase)

            ct = ContextTracker()
            ct.context = fetch
            _, tokens = ct.get_tokens(card)
            assert ngramutil.fetch_like(tokens) == [(expected, 1)]


if __name__ == "__main__":
    unittest.main()
