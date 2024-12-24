# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: GPL-3.0-or-later

import unittest

from parameterized import parameterized

from convassist.utilities.ngram.ngram_map import NgramMap


class TestNgramMap(unittest.TestCase):
    def setUp(self) -> None:
        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()

    def testCreateNgramMap(self):
        phrase = "all your bases are mine"
        cardinality = 3
        expected = [
            (["all", "your", "bases"], 1),
            (["your", "bases", "are"], 1),
            (["bases", "are", "mine"], 1),
        ]

        ngram_map = NgramMap(cardinality, phrase)

        self.assertEqual(len(ngram_map), 3)
        self.assertEqual(ngram_map.all_items(), expected)

    @parameterized.expand(
        [
            ("all your bases are mine", 3),
            ("all your bases are mine", 2),
            ("all your bases are mine", 1),
        ]
    )
    def testIdenticalNgramMaps(self, phrase, card):
        phrase = "all your bases are mine"

        ngram_map1 = NgramMap(card, phrase)
        ngram_map2 = NgramMap(card, phrase)

        self.assertEqual(ngram_map1.cardinality, card)
        self.assertEqual(ngram_map2.cardinality, card)
        self.assertListEqual(ngram_map1.all_items(), ngram_map2.all_items())
