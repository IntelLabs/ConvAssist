# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: GPL-3.0-or-later

from nltk import RegexpTokenizer

#TODO: Investigate using the SpellChecker to improve the context tracker

class ContextTracker:
    """
    Tracks the current context.

    """

    def __init__(self, lowercase_mode=False):
        self.lowercase = lowercase_mode
        self.tokens = []
        self._context = ""

    def _update_context(self):
        # tokenize the context into words with punctuation and spaces
        # reg = r"\w+(?:['-]\w+)*|\s"  # with spaces
        reg = r"\w+(?:['-]\w+)*"  # without spaces

        tokenizer = RegexpTokenizer(reg)
        self.tokens = tokenizer.tokenize(self._context.lower())

        if self._context and self._context[-1] == " ":  # if the last character is a space
            self.tokens.append("")

    def token(self, index):

        if 0 <= index < len(self.tokens):
            return self.tokens[index]
        else:
            return ""

    def get_tokens(self, count: int):
        actual_tokens = count if len(self.tokens) >= count else len(self.tokens)
        return actual_tokens, self.tokens[-actual_tokens:]

    def get_last_token(self):
        return self.tokens[-1] if len(self.tokens) > 0 else ""

    @property
    def context(self):
        return self._context

    @context.setter
    def context(self, value):
        self._context = value.lower() if value else value
        self._update_context()

    @property
    def token_count(self):
        return len(self.tokens)
