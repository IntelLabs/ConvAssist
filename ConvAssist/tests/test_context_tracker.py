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
        tokens = [""] * 4
        count, tokens = tracker.get_tokens(len(tokens))
        assert count == 4
        self.assertEqual(tokens, ["example", " ", "token", " "])

    def test_get_tokens_empty(self):
        tracker = ContextTracker()
        tokens = [""] * 2
        count, tokens = tracker.get_tokens(len(tokens))
        assert count == 0
        self.assertEqual(tokens, ["", ""])

    def test_context_has_punctuation(self):
        tracker = ContextTracker()
        tracker.context = "example! token, with %puncuation's."
        tokens = [""] * 4
        count, tokens = tracker.get_tokens(len(tokens))
        assert count == 4
        self.assertEqual(tokens, ["example", "token", "with", "puncuation's"])

if __name__ == '__main__':
    unittest.main()