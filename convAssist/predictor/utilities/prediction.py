# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

MIN_PROBABILITY = 0.0
MAX_PROBABILITY = 1.0

class UnknownCombinerException(Exception):
    pass

class Prediction(list):
    """
    Class for predictions from predictors.

    """

    def __init__(self):
        pass

    def __eq__(self, other):
        if self is other:
            return True
        if len(self) != len(other):
            return False
        for i, s in enumerate(other):
            if not s == self[i]:
                return False
        return True

    def suggestion_for_token(self, token):
        for s in self:
            if s.word == token:
                return s

    def add_suggestion(self, suggestion):
        if len(self) == 0:
            self.append(suggestion)
        else:
            i = 0
            while i < len(self) and suggestion < self[i]:
                i += 1

            self.insert(i, suggestion)