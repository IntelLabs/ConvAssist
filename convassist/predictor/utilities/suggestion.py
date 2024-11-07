# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: GPL-3.0-or-later

MIN_PROBABILITY = 0.0
MAX_PROBABILITY = 1.0


class SuggestionException(Exception):
    pass


class Suggestion:
    """
    Class for a simple suggestion, consists of a string and a probility for that
    string.

    """

    def __init__(self, word: str, probability: float, predictor_name: str):
        self._word = word
        self._probability = probability
        self._predictor_name = predictor_name

    def __eq__(self, other):
        if self.word == other.word and self.probability == other.probability:
            return True
        return False

    def __lt__(self, other):
        if self.probability < other.probability:
            return True
        if self.probability == other.probability:
            return self.word < other.word
        return False

    def __repr__(self):
        return f"Suggestion: {self.word} - Probability: {self.probability}"

    @property
    def word(self):
        return self._word

    @property
    def predictor_name(self):
        return self._predictor_name

    @property
    def probability(self):
        return self._probability

    @probability.setter
    def probability(self, value):
        # if value < MIN_PROBABILITY or value > MAX_PROBABILITY:
        #     raise SuggestionException("Probability is too high or too low = " + str(value))

        self._probability = value

    @probability.deleter
    def probability(self):
        del self._probability
