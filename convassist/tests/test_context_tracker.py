# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: GPL-3.0-or-later

import unittest
from configparser import ConfigParser
from unittest.mock import MagicMock

from ..context_tracker import ContextTracker


class TestContextTracker(unittest.TestCase):
    def test_token(self):
        tracker = ContextTracker()
        tracker.context = "example token "
        token = tracker.token(0)
        self.assertEqual(token, "example")

    def test_n_token(self):
        tracker = ContextTracker()
        tracker.context = "example token "
        token = tracker.token(1)
        self.assertEqual(token, "token")

    def test_get_tokens(self):
        tracker = ContextTracker()
        tracker.context = "example token"
        count, tokens = tracker.get_tokens(2)
        assert count == 2
        self.assertEqual(tokens, ["example", "token"])

    def test_get_tokens_empty(self):
        tracker = ContextTracker()
        count, tokens = tracker.get_tokens(1)
        assert count == 0
        self.assertEqual(tokens, [])

    def test_context_has_punctuation(self):
        tracker = ContextTracker()
        tracker.context = "tokens with punctuation's and hypenated-words "
        count, tokens = tracker.get_tokens(10)
        assert count == 6
        self.assertEqual(tokens, ["tokens", "with", "punctuation's", "and", "hypenated-words", ""])

    def test_get_last_token(self):
        tracker = ContextTracker()
        tracker.context = "example token"
        last_token = tracker.get_last_token()
        self.assertEqual(last_token, "token")

    def test_get_last_token_with_space(self):
        tracker = ContextTracker()
        tracker.context = "example token "
        last_token = tracker.get_last_token()
        self.assertEqual(last_token, "")

    def test_get_last_token_empty(self):
        tracker = ContextTracker()
        last_token = tracker.get_last_token()
        self.assertEqual(last_token, "")

    def test_context_setter(self):
        tracker = ContextTracker()
        tracker.context = "new context"
        self.assertEqual(tracker.context, "new context")
        self.assertEqual(tracker.tokens, ["new", "context"])

    def test_context_setter_empty(self):
        tracker = ContextTracker()
        tracker.context = ""
        self.assertEqual(tracker.context, "")
        self.assertEqual(tracker.tokens, [])

    def test_context_tracker_token_beyond_len(self):
        tracker = ContextTracker()
        tracker.context = "example token"
        token = tracker.token(2)
        self.assertEqual(token, "")

    def test_context_tracker_token_negative_index(self):
        tracker = ContextTracker()
        tracker.context = "example token"
        token = tracker.token(-1)
        self.assertEqual(token, "")

    def test_context_tracker_token_empty_context(self):
        tracker = ContextTracker()
        tracker.context = ""
        token = tracker.token(0)
        self.assertEqual(token, "")

    def test_get_tokens_space_context(self):
        tracker = ContextTracker()
        tracker.context = " "
        count, tokens = tracker.get_tokens(1)
        assert count == 1
        self.assertEqual(tokens, [""])


if __name__ == "__main__":
    unittest.main()
