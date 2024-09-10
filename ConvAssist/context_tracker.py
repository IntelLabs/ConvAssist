# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""
Class for context tracker.

"""
# from nltk.tokenize import word_tokenize
from nltk import WordPunctTokenizer, RegexpTokenizer

class ContextTracker(object):
    """
    Tracks the current context.

    """

    def __init__(self, lowercase_mode=False):
        self.lowercase = lowercase_mode
        self.tokens = []
        self._context = ""

    def _update_context(self):
        reg = r"\w+|\s+"
        tokenizer = RegexpTokenizer(reg)
        # tokenizer = WordPunctTokenizer()
        self.tokens = tokenizer.tokenize(self._context)
        # self.tokens.reverse()

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