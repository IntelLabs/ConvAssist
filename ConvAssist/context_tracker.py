# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: GPL-3.0-or-later

from nltk import RegexpTokenizer


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
        self.tokens = tokenizer.tokenize(self._context)

    def token(self, index):

        if 0 <= index < len(self.tokens):
            return self.tokens[index]
        else:
            return ""

    def get_tokens(self, count: int):
        # TODO: FIXME Edge case - if there is only one token and it's a space return 0 tokens
        if len(self.tokens) == 1 and self.tokens[0] == " ":
            return 0, []

        actual_tokens = count if len(self.tokens) >= count else len(self.tokens)
        return actual_tokens, self.tokens[:actual_tokens]

    def get_last_token(self):
        # filter out the last token if it is a space and return the next token
        if self.tokens:
            tmp = list(filter(lambda x: x != " ", self.tokens))
            return tmp[-1] if tmp else ""
        else:
            return ""

        # tmp = self.tokens.filter(lambda x: x != " ")

        # last_token = self.tokens[-1] if len(self.tokens) > 0 else ""

        # #TODO: FIXME - if the last token is a space, return an empty string
        # return last_token if last_token != " " else ""

    @property
    def context(self):
        return self._context

    @context.setter
    def context(self, value):
        self._context = value
        self._update_context()
