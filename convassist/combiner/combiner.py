# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: GPL-3.0-or-later

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: GPL-3.0-or-later
"""
Combiner classes to merge results from several predictors.
"""
import abc
from typing import Set

from convassist.predictor.utilities.prediction import Prediction
from convassist.predictor.utilities.suggestion import MAX_PROBABILITY


class Combiner(metaclass=abc.ABCMeta):
    """
    Base class for all combiners
    """

    def filter(self, prediction: Prediction) -> Prediction:
        seen_tokens: Set[str] = set()
        result = Prediction()
        for i, suggestion in enumerate(prediction):
            token = suggestion.word
            if token not in seen_tokens:
                for j in range(i + 1, len(prediction)):
                    if token == prediction[j].word:
                        # TODO: interpolate here?
                        suggestion.probability += prediction[j].probability
                        if suggestion.probability > MAX_PROBABILITY:
                            suggestion.probability = MAX_PROBABILITY
                seen_tokens.add(token)
                result.add_suggestion(suggestion)
        return result

    @abc.abstractmethod
    def combine(self):
        raise NotImplementedError("Method must be implemented")  # pragma: no cover
