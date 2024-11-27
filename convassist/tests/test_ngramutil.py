# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: GPL-3.0-or-later

import unittest

from parameterized import parameterized

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

            assert ngramutil._table_exists(f"_{self.cardinality}_gram")

    def test_learn(self):
        # Test the learn method of NGramUtil
        with NGramUtil(":memory:", self.cardinality) as ngramutil:
            assert ngramutil._table_exists(f"_{self.cardinality}_gram")

            ngramutil.learn("test")
            assert ngramutil.unigram_counts_sum() == 1

    def test_learn_updates(self):
        # Test the learn method of NGramUtil
        with NGramUtil(":memory:", self.cardinality) as ngramutil:
            assert ngramutil._table_exists(f"_{self.cardinality}_gram")

            ngramutil.learn("test")
            assert ngramutil.unigram_counts_sum() == 1

            ngramutil.learn("test")
            assert ngramutil.unigram_counts_sum() == 2

    def test_fetch_like(self):
        # Test the fetch_like method of NGramUtil
        self.cardinality = 1
        with NGramUtil(":memory:", self.cardinality) as ngramutil:
            assert ngramutil._table_exists(f"_{self.cardinality}_gram")

            ngramutil.learn("test")
            assert ngramutil.unigram_counts_sum() == 1

            ngramutil.learn("test")
            assert ngramutil.unigram_counts_sum() == 2

            assert ngramutil.fetch_like(["test"], 1) == [("test", 2)]

    def test_fetch_like_multiple(self):
        # Test the fetch_like method of NGramUtil
        self.cardinality = 1
        # database = ":memory:"
        database = "test.db"
        with NGramUtil(database, self.cardinality) as ngramutil:
            assert ngramutil._table_exists(f"_{self.cardinality}_gram")

            ngramutil.learn("test")
            assert ngramutil.unigram_counts_sum() == 1

            ngramutil.learn("test")
            assert ngramutil.unigram_counts_sum() == 2

            ngramutil.learn("test")
            assert ngramutil.unigram_counts_sum() == 3

            assert ngramutil.fetch_like(["test"], 2) == [("test", 3)]

    @parameterized.expand(
        [
            ("all your bases are mine", "bases are", "mine", 3),
            ("all your bases are mine", "are", "mine", 2),
            ("all your bases are mine", "mine", "mine", 1),
            ("it's a beautiful day in the neighborhood", "beautiful day", "in", 3),
            ("it's beautiful", "it's", "beautiful", 2),
            ("it's", "it's", "it's", 1),
        ]
    )
    def test_fetch_list_multi_ngram(self, phrase, fetch, expected, card):
        database = ":memory:"
        # database = "test.db"
        with NGramUtil(database, card) as ngramutil:
            assert ngramutil._table_exists(f"_{card}_gram")

            ngramutil.learn(phrase)

            assert ngramutil.fetch_like(str.split(fetch), 1) == [(expected, 1)]


if __name__ == "__main__":
    unittest.main()
