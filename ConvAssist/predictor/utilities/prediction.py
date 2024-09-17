# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: GPL-3.0-or-later

MIN_PROBABILITY = 0.0
MAX_PROBABILITY = 1.0

from ConvAssist.predictor.utilities.suggestion import Suggestion

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

    def add_suggestion(self, suggestion:Suggestion):
        self.append(suggestion)
        self.sort(key=lambda x: x.probability, reverse=True)
