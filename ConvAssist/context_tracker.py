# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: GPL-3.0-or-later


"""
Class for context tracker.

"""
from nltk import RegexpTokenizer

class ContextTracker(object):
    """
    Tracks the current context.

    """

    def __init__(self, lowercase_mode=False):
        self.lowercase = lowercase_mode
        self.tokens = []
        self._context = ""

    def _update_context(self):
        # tokenize the context into words with punctuation and spaces
        # reg = r"\w+('\w+)?|\w+(-\w+)?|\s+"
        # reg = r'\w+(?:-\w+)*|\s|\w+(?:\'\w+)?'
        reg = r"\w+(?:['-]\w+)*|\s"
        tokenizer = RegexpTokenizer(reg)
        self.tokens = tokenizer.tokenize(self._context)

    def token(self, index):

        if 0 <= index < len(self.tokens):
            return self.tokens[index]
        else:
            return ""
        
    def get_tokens(self, count: int):
        actual_tokens = count if len(self.tokens) >= count else len(self.tokens)   
        return actual_tokens, self.tokens[-count:]

    def get_last_token(self):
        return self.tokens[-1] if len(self.tokens) > 0 else ""

    @property
    def context(self):
        return self._context
    
    @context.setter
    def context(self, value):
        self._context = value
        self._update_context()