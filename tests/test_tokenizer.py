# Copyright (C) <year(s)> Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import unittest
from pathlib import Path

from  src.tokenizer.forward_tokenizer import ForwardTokenizer
from src.tokenizer.reverse_tokenizer import ReverseTokenizer


class TestForwardTokenizer(unittest.TestCase):
    
    def setUp(self):
        filename = self.filename = Path("tests/test_data") / ("der_linksdenker.txt")
        with open(filename, "r", encoding="utf-8") as f:
            self.tokenizer = ForwardTokenizer(f.read())

    def test_reset_stream(self):
        self.tokenizer.next_token()
        assert self.tokenizer.offset != 0
        self.tokenizer.reset_stream()
        assert self.tokenizer.offset == 0

    def test_count_characters(self):
        assert self.tokenizer.count_characters() == 7927

    def test_count_tokens(self):
        assert self.tokenizer.count_tokens() == 1251

    def test_has_more_tokens(self):
        assert self.tokenizer.has_more_tokens() == True

    def test_next_token(self):
        assert self.tokenizer.next_token() == "Der"
        self.tokenizer.reset_stream()

    def test_tokenization(self):
        tokenizer = ForwardTokenizer("This is ConvAssist.")
        tokens = list(tokenizer)
        self.assertEqual(tokens, ["This", "is", "ConvAssist"])

    def test_is_blankspace(self):
        assert self.tokenizer.is_blankspace("\n") == True
        assert self.tokenizer.is_blankspace("a") == False

    def test_is_separator(self):
        assert self.tokenizer.is_separator('"') == True
        assert self.tokenizer.is_separator("b") == False


class TestReverseTokenizer(unittest.TestCase):
    def setUp(self):
        filename = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "test_data", "der_linksdenker.txt")
        )
        with open(filename, "r", encoding="utf-8") as f:
            self.tokenizer = ReverseTokenizer(f.read())

    def test_reset_stream(self):
        self.tokenizer.next_token()
        assert self.tokenizer.offset != self.tokenizer.offend
        self.tokenizer.reset_stream()
        assert self.tokenizer.offset == self.tokenizer.offend

    def test_count_tokens(self):
        assert self.tokenizer.count_tokens() == 1251

    def test_has_more_tokens(self):
        assert self.tokenizer.has_more_tokens() == True

    def test_next_token(self):
        assert self.tokenizer.next_token() == "Linksdenker"
        self.tokenizer.reset_stream()


class TestEqual(unittest.TestCase):
    def test_tokenizers_are_equal(self):
        filename = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "test_data", "der_linksdenker.txt")
        )
        with open(filename, "r", encoding="utf-8") as f:
            reverse_tokenizer = ReverseTokenizer(f.read())
            f.seek(0)
            forward_tokenizer = ForwardTokenizer(f.read())

        forward_tokens = []
        reverse_tokens = []
        while forward_tokenizer.has_more_tokens():
            forward_tokens.append(forward_tokenizer.next_token())
        while reverse_tokenizer.has_more_tokens():
            reverse_tokens.append(reverse_tokenizer.next_token())
        diff = set(forward_tokens) ^ set(reverse_tokens)
        assert forward_tokens == reverse_tokens[::-1]
        assert len(diff) == 0
