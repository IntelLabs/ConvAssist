# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: GPL-3.0-or-later

import unittest
from configparser import ConfigParser
from unittest.mock import MagicMock

from ConvAssist.context_tracker import ContextTracker


class TestContextTracker(unittest.TestCase):
    def test_token(self):
        tracker = ContextTracker()
        tracker.context = "example token "
        token = tracker.token(0)
        self.assertEqual(token, "example")

    def test_n_token(self):
        tracker = ContextTracker()
        tracker.context = "example token "
        token = tracker.token(2)
        self.assertEqual(token, "token")

    def test_get_tokens(self):
        tracker = ContextTracker()
        tracker.context = "example token "
        count, tokens = tracker.get_tokens(4)
        assert count == 4
        self.assertEqual(tokens, ["example", " ", "token", " "])

    def test_get_tokens_empty(self):
        tracker = ContextTracker()
        count, tokens = tracker.get_tokens(1)
        assert count == 0
        self.assertEqual(tokens, [])

    def test_context_has_punctuation(self):
        tracker = ContextTracker()
        tracker.context = "tokens with punctuation's and hypenated-words "
        count, tokens = tracker.get_tokens(10)
        assert count == 10
        self.assertEqual(
            tokens,
            ["tokens", " ", "with", " ", "punctuation's", " ", "and", " ", "hypenated-words", " "],
        )


if __name__ == "__main__":
    unittest.main()
