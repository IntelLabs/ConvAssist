# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: GPL-3.0-or-later

import unittest

from convassist.predictor.utilities.suggestion import Suggestion, SuggestionException


class TestSuggestion(unittest.TestCase):
    def test_init(self):
        word = "test"
        probability = 0.5
        predictor_name = "predictor"
        suggestion = Suggestion(word, probability, predictor_name)
        self.assertEqual(suggestion.word, word)
        self.assertEqual(suggestion.probability, probability)
        self.assertEqual(suggestion.predictor_name, predictor_name)

    def test_eq(self):
        suggestion1 = Suggestion("test", 0.5, "predictor")
        suggestion2 = Suggestion("test", 0.5, "predictor")
        suggestion3 = Suggestion("different", 0.5, "predictor")
        self.assertEqual(suggestion1, suggestion2)
        self.assertNotEqual(suggestion1, suggestion3)

    def test_lt(self):
        suggestion1 = Suggestion("apple", 0.5, "predictor")
        # suggestion2 = Suggestion("banana", 0.5, "predictor")
        suggestion3 = Suggestion("apple", 0.3, "predictor")
        self.assertLess(suggestion3, suggestion1)
        # self.assertLess(suggestion2, suggestion1)
        self.assertGreater(suggestion1, suggestion3)
        # self.assertGreater(suggestion1, suggestion2)

    def test_repr(self):
        suggestion = Suggestion("test", 0.5, "predictor")
        expected_repr = "Suggestion: test - Probability: 0.5"
        self.assertEqual(repr(suggestion), expected_repr)

    def test_probability_setter_valid(self):
        suggestion = Suggestion("test", 0.5, "predictor")
        suggestion.probability = 0.7
        self.assertEqual(suggestion.probability, 0.7)

    def test_probability_setter_invalid(self):
        suggestion = Suggestion("test", 0.5, "predictor")
        with self.assertRaises(SuggestionException):
            suggestion.probability = 1.5

    def test_probability_deleter(self):
        suggestion = Suggestion("test", 0.5, "predictor")
        del suggestion.probability
        with self.assertRaises(AttributeError):
            _ = suggestion.probability


if __name__ == "__main__":
    unittest.main()
