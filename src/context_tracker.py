# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Class for context tracker.

"""
from src.utilities.character import *
from src.tokenizer.forward_tokenizer import ForwardTokenizer
from src.tokenizer.reverse_tokenizer import ReverseTokenizer
from src.predictor_registry import PredictorRegistry

DEFAULT_SLIDING_WINDOW_SIZE = 80


class InvalidCallbackException(Exception):
    pass


class ContextChangeDetector(object):
    def __init__(self, lowercase):
        self.lowercase = lowercase
        self.sliding_windows_size = DEFAULT_SLIDING_WINDOW_SIZE
        self.sliding_window = ""

    def update_sliding_window(self, string):
        if len(string) <= self.sliding_windows_size:
            self.sliding_window = string
        else:
            self.sliding_window = string[: -self.sliding_windows_size]

    def context_change(self, past_stream):
        # rename for clarity
        prev_context = self.sliding_window
        curr_context = past_stream

        if len(prev_context) == 0:
            if len(curr_context) == 0:
                return False
            else:
                return True

        ctx_idx = curr_context.rfind(prev_context)
        if ctx_idx == -1:
            return True

        remainder = curr_context[ctx_idx + len(prev_context) :]
        idx = last_word_character(remainder)
        if idx == -1:
            if len(remainder) == 0:
                return False
            last_char = curr_context[ctx_idx + len(prev_context) - 1]
            if is_word_character(last_char):
                return False
            else:
                return True

        if idx == len(remainder) - 1:
            return False

        return True

    def change(self, past_stream):
        # rename for clarity
        prev_context = self.sliding_window
        curr_context = past_stream
        if len(prev_context) == 0:
            return past_stream

        ctx_idx = curr_context.rfind(prev_context)
        if ctx_idx == -1:
            return past_stream

        result = curr_context[ctx_idx + len(prev_context) :]
        if self.context_change(past_stream):
            sliding_window_stream = self.sliding_window
            r_tok = ReverseTokenizer(sliding_window_stream)
            r_tok.lowercase = self.lowercase
            first_token = r_tok.next_token()
            if not len(first_token) == 0:
                result = first_token + result

        return result


class ContextTracker(object):
    """
    Tracks the current context.
    """

    def __init__(self, config, predictor_registry, callback):
        self.config = config
        self.lowercase = self.config.getboolean("ContextTracker", "lowercase_mode", fallback=True)

        self.registry: PredictorRegistry = predictor_registry
        if callback:
            self.callback = callback
        else:
            raise InvalidCallbackException

        self.context_change_detector = ContextChangeDetector(self.lowercase)
        self.registry.context_tracker = self

        self.sliding_windows_size = DEFAULT_SLIDING_WINDOW_SIZE

    def context_change(self):
        return self.context_change_detector.context_change(self.past_stream())

    def update_context(self):
        change = self.context_change_detector.change(self.past_stream())
        tok = ForwardTokenizer(change)
        tok.lowercase = self.lowercase

        change_tokens = []
        while tok.has_more_tokens():
            token = tok.next_token()
            change_tokens.append(token)

        if len(change_tokens) != 0:
            # remove prefix (partially entered token or empty token)
            change_tokens.pop()

        for predictor in self.registry:
            if change_tokens:
                predictor.learn(change_tokens)

        self.context_change_detector.update_sliding_window(self.past_stream())

    def prefix(self):
        return self.token(0)

    def token(self, index):
        past_string_stream = self.past_stream()
        past_string_stream = past_string_stream.lstrip()
        tok = ReverseTokenizer(past_string_stream)
        tok.lowercase = self.lowercase
        i = 0
        while tok.has_more_tokens() and i <= index:
            token = tok.next_token()
            i += 1
        if i <= index:
            token = ""

        return token

    def extra_token_to_learn(self, index, change):
        return self.token(index + len(change))

    def future_stream(self):
        return self.callback.future_stream()

    def past_stream(self):
        return self.callback.past_stream()

    def is_completion_valid(self, completion):
        prefix = self.prefix().lower()
        if prefix and prefix in completion:
            return True
        return False

    def __repr__(self):
        return self.callback.past_stream + "<|>" + self.callback.future_stream + "\n"
